"""
nvidia-smi web monitor

Modified from gpustat.web to use nvidia-smi instead of gpustat
"""

from typing import List, Tuple, Optional, Union
import json
import re
import os
import traceback
import urllib.parse
import ssl

import asyncio
import asyncssh
import aiohttp

from datetime import datetime
from collections import OrderedDict

from termcolor import cprint, colored
from aiohttp import web
import aiohttp_jinja2 as aiojinja2


__PATH__ = os.path.abspath(os.path.dirname(__file__))

# Command to get system stats and GPU info
DEFAULT_NVIDIA_SMI_COMMAND = """
# Get CPU usage
cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\\([0-9.]*\\)%* id.*/\\1/" | awk '{printf "%5.1f", 100-$1}');

# Get memory info
mem_total=$(free -g | awk 'NR==2 {printf "%5.1f", $2}');
mem_used=$(free -g | awk 'NR==2 {printf "%5.1f", $3}');

# Print system info without colors
echo "CPU: ${cpu_usage}% | Memory: ${mem_used} / ${mem_total} GB";

# Get GPU info with updated colors
nvidia-smi --query-gpu=index,name,pstate,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw \
--format=csv,noheader,nounits | awk -F, '{ \
temp=sprintf("%3d", $4); \
util=sprintf("%3d", $5); \
mem_used=sprintf("%5d", $6); \
mem_total=sprintf("%5d", $7); \
power=sprintf("%6.2f", $8); \
printf "[%s] %s | %s | \\033[31m%s°C\\033[0m | \\033[32m%s%%\\033[0m | \\033[33m%s\\033[0m / %s MB | \\033[35m%s W\\033[0m\\n", \
$1, $2, $3, temp, util, mem_used, mem_total, power \
}'"""

RE_ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

class Context(object):
    def __init__(self):
        self.host_status = OrderedDict()
        self.interval = 5.0

    def host_set_message(self, hostname: str, msg: str):
        lines = msg.splitlines()
        if len(lines) >= 2:  # If we have both system stats and GPU info
            separator = "=" * 80 + "\n"
            timestamp = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
            formatted_msg = (
                f"{separator}"
                f"{hostname} {timestamp}\n"
                f"{lines[0]}\n"  # System stats line
                f"{''.join(lines[1:])}\n"  # GPU info lines
            )
            self.host_status[hostname] = formatted_msg
        else:
            self.host_status[hostname] = colored(f"({hostname}) ", 'white') + msg + '\n'


context = Context()


async def run_client(hostname: str, exec_cmd: str, *,
                     port=22, verify_host: bool = True,
                     poll_delay=None, timeout=30.0,
                     name_length=None, verbose=False):
    '''An async handler to collect nvidia-smi through a SSH channel.'''
    L = name_length or 0
    if poll_delay is None:
        poll_delay = context.interval

    def _str(data: Union[bytes, str]) -> str:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return data

    async def _loop_body():
        conn_kwargs = dict()
        if not verify_host:
            conn_kwargs['known_hosts'] = None
        async with asyncssh.connect(hostname, port=port, **conn_kwargs) as conn:
            cprint(f"[{hostname:<{L}}] SSH connection established!", attrs=['bold'])

            while True:
                result = await asyncio.wait_for(conn.run(exec_cmd), timeout=timeout)

                now = datetime.now().strftime('%Y/%m/%d-%H:%M:%S.%f')
                if result.exit_status != 0:
                    cprint(f"[{now} [{hostname:<{L}}] Error, exitcode={result.exit_status}", color='red')
                    cprint(_str(result.stderr or ''), color='red')
                    stderr_summary = _str(result.stderr or '').split('\n')[0]
                    context.host_set_message(hostname, colored(f'[exitcode {result.exit_status}] {stderr_summary}', 'red'))
                else:
                    if verbose:
                        cprint(f"[{now} [{hostname:<{L}}] OK from nvidia-smi "
                               f"({len(_str(result.stdout or ''))} bytes)", color='cyan')
                    # Format the output to be similar to gpustat
                    formatted_output = "=" * 80 + "\n"  # Separator line
                    formatted_output += f"{hostname}  " + datetime.now().strftime('%Y/%m/%d %H:%M:%S') + "\n"
                    formatted_output += _str(result.stdout)
                    formatted_output += "\n"  # Extra newline for spacing
                    context.host_status[hostname] = formatted_output

                await asyncio.sleep(poll_delay)

    while True:
        try:
            await _loop_body()
        except asyncio.CancelledError:
            cprint(f"[{hostname:<{L}}] Closed as being cancelled.", attrs=['bold'])
            break
        except (asyncio.TimeoutError) as ex:
            cprint(f"Timeout after {timeout} sec: {hostname}", color='red')
            context.host_set_message(hostname, colored(f"Timeout after {timeout} sec", 'red'))
        except (asyncssh.misc.DisconnectError, asyncssh.misc.ChannelOpenError, OSError) as ex:
            cprint(f"Disconnected : {hostname}, {str(ex)}", color='red')
            context.host_set_message(hostname, colored(str(ex), 'red'))
        except Exception as e:
            cprint(f"[{hostname:<{L}}] {e}", color='red')
            context.host_set_message(hostname, colored(f"{type(e).__name__}: {e}", 'red'))
            cprint(traceback.format_exc())
            raise

        cprint(f"[{hostname:<{L}}] Disconnected, retrying in {poll_delay} sec...", color='yellow')
        await asyncio.sleep(poll_delay)

# Rest of the code remains similar, just updating the command references
async def spawn_clients(hosts: List[str], exec_cmd: str, *,
                        default_port: int, verify_host: bool = True,
                        verbose=False):
    '''Create a set of async handlers, one per host.'''

    def _parse_host_string(netloc: str) -> Tuple[str, Optional[int]]:
        """Parse a connection string (netloc) in the form of `HOSTNAME[:PORT]`
        and returns (HOSTNAME, PORT)."""
        pr = urllib.parse.urlparse('ssh://{}/'.format(netloc))
        assert pr.hostname is not None, netloc
        return (pr.hostname, pr.port)

    try:
        host_names: List[str]
        host_ports: List[int]
        host_names, host_ports = zip(*(_parse_host_string(host) for host in hosts))  # type: ignore

        # initial response
        for hostname in host_names:
            context.host_set_message(hostname, "Loading ...")

        name_length = max(len(hostname) for hostname in host_names)

        # launch all clients parallel
        await asyncio.gather(*[
            run_client(
                hostname, exec_cmd,
                port=port or default_port,
                verify_host=verify_host,
                verbose=verbose, name_length=name_length
            )
            for (hostname, port) in zip(host_names, host_ports)
        ])
    except Exception as ex:
        # TODO: throw the exception outside and let aiohttp abort startup
        traceback.print_exc()
        cprint(colored("Error: An exception occured during the startup.", 'red'))
        
###############################################################################
# webserver handlers.
###############################################################################

# monkey-patch ansi2html scheme. TODO: better color codes
import ansi2html.style
scheme = 'solarized'
ansi2html.style.SCHEME[scheme] = list(ansi2html.style.SCHEME[scheme])
ansi2html.style.SCHEME[scheme][0] = '#555555'
ansi_conv = ansi2html.Ansi2HTMLConverter(dark_bg=True, scheme=scheme)


def render_gpustat_body(
    mode='html',   # mode: Literal['html'] | Literal['html_full'] | Literal['ansi']
    *,
    full_html: bool = False,
    nodes: Optional[List[str]] = None,
):
    body = ''
    for host, status in context.host_status.items():
        if not status:
            continue
        if nodes is not None and host not in nodes:
            continue
        body += status

    if mode == 'html':
        return ansi_conv.convert(body, full=full_html)
    elif mode == 'ansi':
        return body
    elif mode == 'plain':
        return RE_ANSI.sub('', body)
    else:
        raise ValueError(mode)


async def handler(request):
    '''Renders the html page.'''

    data = dict(
        ansi2html_headers=ansi_conv.produce_headers().replace('\n', ' '),
        http_host=request.host,
        interval=int(context.interval * 1000)
    )
    response = aiojinja2.render_template('index.html', request, data)
    response.headers['Content-Language'] = 'en'
    return response


def _parse_querystring_list(value: Optional[str]) -> Optional[List[str]]:
    return value.strip().split(',') if value else None


def make_static_handler(content_type: str):

    async def handler(request: web.Request):
        # query string handling
        full: bool = request.query.get('full', '1').lower() in ("yes", "true", "1")
        nodes: Optional[List[str]] = _parse_querystring_list(request.query.get('nodes'))

        body = render_gpustat_body(mode=content_type,
                                   full_html=full,
                                   nodes=nodes)
        response = web.Response(body=body)
        response.headers['Content-Language'] = 'en'
        response.headers['Content-Type'] = f'text/{content_type}; charset=utf-8'
        return response

    return handler


async def websocket_handler(request):
    print("INFO: Websocket connection from {} established".format(request.remote))

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    async def _handle_websocketmessage(msg):
        if msg.data == 'close':
            await ws.close()
        else:
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError:
                cprint(f"Malformed message from {request.remote}", color='yellow')
                return

            # send the rendered HTML body as a websocket message.
            nodes: Optional[List[str]] = _parse_querystring_list(payload.get('nodes'))
            body = render_gpustat_body(mode='html', full_html=False, nodes=nodes)
            await ws.send_str(body)

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.CLOSE:
            break
        elif msg.type == aiohttp.WSMsgType.TEXT:
            await _handle_websocketmessage(msg)
        elif msg.type == aiohttp.WSMsgType.ERROR:
            cprint("Websocket connection closed with exception %s" % ws.exception(), color='red')

    print("INFO: Websocket connection from {} closed".format(request.remote))
    return ws


def create_app(*,
               hosts=['localhost'],
               default_port: int = 22,
               verify_host: bool = True,
               ssl_certfile: Optional[str] = None,
               ssl_keyfile: Optional[str] = None,
               exec_cmd: Optional[str] = None,
               verbose=True):
    if not exec_cmd:
        exec_cmd = DEFAULT_NVIDIA_SMI_COMMAND

    app = web.Application()
    app.router.add_get('/', handler)
    app.add_routes([web.get('/ws', websocket_handler)])
    app.add_routes([web.get('/nvidia-smi.html', make_static_handler('html'))])
    app.add_routes([web.get('/nvidia-smi.ansi', make_static_handler('ansi'))])
    app.add_routes([web.get('/nvidia-smi.txt', make_static_handler('plain'))])

    async def start_background_tasks(app):
        clients = spawn_clients(
            hosts, exec_cmd, default_port=default_port,
            verify_host=verify_host,
            verbose=verbose)
        loop = app.loop if hasattr(app, 'loop') else asyncio.get_event_loop()
        app['tasks'] = loop.create_task(clients)
        await asyncio.sleep(0.1)
    app.on_startup.append(start_background_tasks)

    async def shutdown_background_tasks(app):
        cprint(f"... Terminating the application", color='yellow')
        app['tasks'].cancel()
    app.on_shutdown.append(shutdown_background_tasks)

    # jinja2 setup
    import jinja2
    aiojinja2.setup(app,
                    loader=jinja2.FileSystemLoader(
                        os.path.join(__PATH__, 'template'))
                    )

    # SSL setup
    if ssl_certfile and ssl_keyfile:
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(certfile=ssl_certfile,
                                    keyfile=ssl_keyfile)
        cprint(f"Using Secure HTTPS (SSL/TLS) server ...", color='green')
    else:
        ssl_context = None
    return app, ssl_context


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hosts', nargs='*',
                        help='List of nodes. Syntax: HOSTNAME[:PORT]')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--port', type=int, default=48109,
                        help="Port number the web application will listen to. (Default: 48109)")
    parser.add_argument('--ssh-port', type=int, default=22,
                        help="Default SSH port to establish connection through. (Default: 22)")
    parser.add_argument('--no-verify-host', action='store_true',
                        help="Skip SSH Host key verification. SSH host verification is turned on by default.")
    parser.add_argument('--interval', type=float, default=5.0,
                        help="Interval (in seconds) between two consecutive requests.")
    parser.add_argument('--ssl-certfile', type=str, default=None,
                        help="Path to the SSL certificate file (Optional, if want to run HTTPS server)")
    parser.add_argument('--ssl-keyfile', type=str, default=None,
                        help="Path to the SSL private key file (Optional, if want to run HTTPS server)")
    parser.add_argument('--exec', type=str,
                        default=DEFAULT_NVIDIA_SMI_COMMAND,
                        help="command-line to execute (default: nvidia-smi with formatted output)")
    args = parser.parse_args()

    hosts = args.hosts or ['localhost']
    cprint(f"Hosts : {hosts}", color='green')
    cprint(f"Cmd   : {args.exec}", color='yellow')

    if args.interval > 0.1:
        context.interval = args.interval

    app, ssl_context = create_app(
        hosts=hosts, default_port=args.ssh_port,
        verify_host=not args.no_verify_host,
        ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile,
        exec_cmd=args.exec,
        verbose=args.verbose)

    web.run_app(app, host='0.0.0.0', port=args.port,
                ssl_context=ssl_context)

if __name__ == '__main__':
    main()
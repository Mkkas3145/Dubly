"""Port configuration and loopback-only control for the bundled desktop launcher."""
import hmac,os
from fastapi import APIRouter,HTTPException,Request

_shutdown=None


def port(environ=None):
    raw=(os.environ if environ is None else environ).get('DUBLY_PORT','20806')
    try:value=int(raw)
    except (TypeError,ValueError):raise ValueError('Port must be a number between 1 and 65535.') from None
    if not 1<=value<=65535:raise ValueError('Port must be a number between 1 and 65535.')
    return value


def _owner(request):
    token=os.environ.get('DUBLY_LAUNCHER_TOKEN','')
    supplied=request.headers.get('authorization','').removeprefix('Bearer ')
    local=request.client and request.client.host in ('127.0.0.1','::1')
    return bool(local and token and hmac.compare_digest(token.encode('utf8'),supplied.encode('utf8')))


def router():
    routes=APIRouter()
    @routes.get('/api/launcher/health')
    def health(request:Request):
        owner=_owner(request)
        return {'service':'dubly','api_version':1,'managed':owner,'pid':os.getpid() if owner else None}
    @routes.post('/api/launcher/shutdown')
    async def shutdown(request:Request):
        if not _owner(request) or _shutdown is None:raise HTTPException(403,'Only the server started by this launcher can be stopped.')
        import asyncio
        asyncio.get_running_loop().call_later(.1,_shutdown)
        return {'ok':True}
    return routes


def run(app):
    global _shutdown
    import uvicorn
    server=uvicorn.Server(uvicorn.Config(app,host='0.0.0.0',port=port()))
    _shutdown=lambda:setattr(server,'should_exit',True)
    server.run()

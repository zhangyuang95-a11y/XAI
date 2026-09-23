"""LOCAL ONLY: synthetic preview sessions; no Prolific or provider access."""
import sys,json,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study_v3.server import make_server,COOKIE
from study_v3.config import Settings
from study_v3.registry import engine
from study_v3.store import StudyError

ROOT=Path(__file__).resolve().parents[1]
settings=Settings(database=str(ROOT/'output/automatic-explanations-preview.sqlite3'),
    automatic_explanations=True,admin_token='local-automatic-preview-only',origin='http://127.0.0.1:9130',mode='preview')
server=make_server(settings,port=9130)
store=server.store
def prepare(domain, existing_token=None):
    name='auto-preview-'+uuid.uuid4().hex[:8]
    if existing_token:
        try:
            with store.db.transaction(read_only=True) as db:
                participant=store._participant(db,existing_token)
            if participant['id'].startswith('auto-preview-'):name=participant['id']
            else:existing_token=None
        except StudyError:existing_token=None
    token,view=store.create({'participant_id':name,
        'mode':'preview','domain':domain,'group':'A','language':'en','consent':True},token=existing_token,admin=True)
    if view['stage']!='demo':return token
    def command(kind,**fields):
        nonlocal view
        view=store.command(token,kind,{'instance_id':view['instance_id'],'revision':view['revision'],'command_id':uuid.uuid4().hex,**fields})
    command('demo_skip')
    # Bypass baseline only in this isolated preview DB, to show the requested UI.
    with store.db.transaction() as db:
        db.execute("UPDATE pl3_runs SET status='completed' WHERE id=?",(view['run_id'],))
    command('next')
    for _ in range(100):
        if view['automatic_explanations'] or view['state']['terminal']:break
        with store.db.transaction(read_only=True) as db:
            state=json.loads(db.one('SELECT state_json FROM pl3_runs WHERE id=?',(view['run_id'],))['state_json'])
        action='wait' if domain=='kitchen' else engine(domain).human_advisor(state)
        command('action',run_id=view['run_id'],turn=state['turn'],action=action)
    print(domain,'preview prepared; turn',view['state']['turn'],'cards',len(view['automatic_explanations']),flush=True)
    return token
original=server.RequestHandlerClass
class PreviewHandler(original):
    def do_GET(self):
        domain=self.path.removeprefix('/auto-preview/').strip('/')
        if self.path.startswith('/auto-preview/') and domain in ('kitchen','pong','warehouse'):
            token=prepare(domain,self.token())
            self.send_response(302)
            self.send_header('Location','/'+domain+'/')
            self.send_header('Set-Cookie',f'{COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax')
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length','0')
            self.end_headers();return
        super().do_GET()
server.RequestHandlerClass=PreviewHandler
print('LOCAL PREVIEW READY http://127.0.0.1:9130/auto-preview/kitchen',flush=True)
server.serve_forever()

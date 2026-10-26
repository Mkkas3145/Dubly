"""Capability-code collections; video and edit data remain on the same server."""
import hashlib,re,secrets,time,uuid
from fastapi import APIRouter,Body,HTTPException


class SharedProjects:
    def __init__(self,store):
        self.store=store
        with store.connection() as db:
            db.execute('CREATE TABLE IF NOT EXISTS shared_projects (code_hash TEXT PRIMARY KEY,created REAL NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS shared_project_items (code_hash TEXT NOT NULL,video_id TEXT NOT NULL,added REAL NOT NULL,PRIMARY KEY(code_hash,video_id))')

    @staticmethod
    def ids(values):
        if not isinstance(values,list) or len(values)>200:raise ValueError('Invalid video selection for sharing.')
        result=[]
        for value in values:
            if not isinstance(value,str):raise ValueError('Invalid video selection for sharing.')
            try:identifier=str(uuid.UUID(value))
            except ValueError:raise ValueError('Invalid video selection for sharing.') from None
            if identifier not in result:result.append(identifier)
        return result

    @staticmethod
    def code_key(code):
        if not isinstance(code,str):raise ValueError('Invalid sharing code.')
        code=code.strip()
        if not re.fullmatch(r'DUBLY-[A-Za-z0-9_-]{32}',code):raise ValueError('Invalid sharing code.')
        return code,hashlib.sha256(code.encode()).hexdigest()

    def create(self,ids):
        ids=self.ids(ids);code='DUBLY-'+secrets.token_urlsafe(24);_,key=self.code_key(code)
        with self.store.connection() as db:
            db.execute('INSERT INTO shared_projects VALUES (?,?)',(key,time.time()))
        return {'code':code,**self.sync(code,ids)}

    def sync(self,code,ids):
        _,key=self.code_key(code);ids=self.ids(ids)
        with self.store.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            if not db.execute('SELECT 1 FROM shared_projects WHERE code_hash=?',(key,)).fetchone():raise KeyError(code)
            if ids:
                known=db.execute('SELECT id FROM videos WHERE id IN ('+','.join('?'*len(ids))+') AND state!=?',(*ids,'deleting')).fetchall()
                db.executemany('INSERT OR IGNORE INTO shared_project_items VALUES (?,?,?)',[(key,row['id'],time.time()) for row in known])
            # Use the existing history capacity and retain the newest projects.
            db.execute('DELETE FROM shared_project_items WHERE code_hash=? AND video_id NOT IN (SELECT i.video_id FROM shared_project_items i JOIN videos v ON v.id=i.video_id WHERE i.code_hash=? ORDER BY v.created DESC LIMIT 200)',(key,key))
            rows=db.execute('SELECT v.id FROM shared_project_items i JOIN videos v ON v.id=i.video_id WHERE i.code_hash=? AND v.state!=? ORDER BY v.created DESC,v.id',(key,'deleting')).fetchall()
        return {'ids':[r['id'] for r in rows]}


def router(api):
    routes=APIRouter()
    @routes.post('/api/project-share/create')
    def create(payload:dict=Body(...)):
        try:return SharedProjects(api.store).create(payload.get('ids',[]))
        except ValueError as error:raise HTTPException(400,str(error)) from None
    @routes.post('/api/project-share/sync')
    def sync(payload:dict=Body(...)):
        try:return SharedProjects(api.store).sync(payload.get('code'),payload.get('ids',[]))
        except ValueError as error:raise HTTPException(400,str(error)) from None
        except KeyError:raise HTTPException(404,'Sharing code not found. Verify that it belongs to this website.') from None
    return routes

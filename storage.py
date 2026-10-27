from __future__ import annotations
from contextlib import contextmanager
import sqlite3
import shutil
import time
import uuid
from core import ROOT,SOURCE_SELECTIONS,TARGET_LANGUAGES


class Store:
    def __init__(self, root=ROOT):
        self.root = root
        (root / 'work').mkdir(parents=True, exist_ok=True)
        self.path = root / 'work' / 'queue.sqlite3'
        with self.connection() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('''CREATE TABLE IF NOT EXISTS videos (
                position INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE NOT NULL, filename TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'queued', phase TEXT NOT NULL DEFAULT '대기',
                phase_progress REAL NOT NULL DEFAULT 0, progress REAL NOT NULL DEFAULT 0,
                detail TEXT NOT NULL DEFAULT '', error TEXT, created REAL NOT NULL, updated REAL NOT NULL
            )''')
            columns={r['name'] for r in db.execute('PRAGMA table_info(videos)')}
            for field in ('source_language','target_language','detected_source_language','translation_prompt'):
                if field not in columns:db.execute(f'ALTER TABLE videos ADD COLUMN {field} TEXT')
            if 'workflow_action' not in columns:db.execute("ALTER TABLE videos ADD COLUMN workflow_action TEXT NOT NULL DEFAULT 'full'")
            if 'resource_state' not in columns:db.execute("ALTER TABLE videos ADD COLUMN resource_state TEXT NOT NULL DEFAULT 'local'")
            if 'editor_index' not in columns:db.execute('ALTER TABLE videos ADD COLUMN editor_index INTEGER')
            if 'tts_preset' not in columns:db.execute("ALTER TABLE videos ADD COLUMN tts_preset TEXT NOT NULL DEFAULT 'creative'")
            if 'tts_sampling_options' not in columns:db.execute('ALTER TABLE videos ADD COLUMN tts_sampling_options TEXT')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def recover(self):
        with self.connection() as db:
            db.execute("UPDATE videos SET state='queued',phase='재개 대기',detail='',updated=? WHERE state='running'", (time.time(),))
            db.execute("UPDATE videos SET state='error',phase='삭제 실패',error='삭제가 중단되었습니다. 다시 삭제해 주세요.',updated=? WHERE state='deleting'", (time.time(),))

    def add(self, video_id, filename, *, source_language=None, target_language=None,tts_preset='creative',translation_prompt=''):
        from tts_presets import options
        options(tts_preset)
        if not isinstance(translation_prompt,str) or len(translation_prompt)>8000:raise ValueError('Additional translation instructions must be 8000 characters or fewer.')
        translation_prompt=translation_prompt.strip()
        if (source_language is None)!=(target_language is None):raise ValueError('Both languages are required')
        if source_language is not None and (source_language not in SOURCE_SELECTIONS or target_language not in TARGET_LANGUAGES or source_language==target_language):
            raise ValueError('Invalid language pair')
        now=time.time()
        with self.connection() as db:
            db.execute('INSERT INTO videos(id,filename,created,updated,source_language,target_language,tts_preset,translation_prompt) VALUES(?,?,?,?,?,?,?,?)',
                       (video_id,filename,now,now,source_language,target_language,tts_preset,translation_prompt))
        return self.get(video_id)

    def add_uploaded(self,items,*,source_language=None,target_language=None,tts_preset='creative',translation_prompt=''):
        from tts_presets import options
        options(tts_preset)
        if not isinstance(translation_prompt,str) or len(translation_prompt)>8000:raise ValueError('Additional translation instructions must be 8000 characters or fewer.')
        translation_prompt=translation_prompt.strip()
        if (source_language is None)!=(target_language is None):raise ValueError('Both languages are required')
        if source_language is not None and (source_language not in SOURCE_SELECTIONS or target_language not in TARGET_LANGUAGES or source_language==target_language):raise ValueError('Invalid language pair')
        now=time.time()
        with self.connection() as db:
            db.executemany("INSERT INTO videos(id,filename,created,updated,source_language,target_language,tts_preset,translation_prompt,state,phase,detail) VALUES(?,?,?,?,?,?,?,?,'uploaded','업로드 완료','분석 준비 완료')",
                [(item['id'],item['filename'],now,now,source_language,target_language,tts_preset,translation_prompt) for item in items])
        return [self.get(item['id']) for item in items]

    def get(self, video_id):
        with self.connection() as db:
            row=db.execute('SELECT * FROM videos WHERE id=?',(video_id,)).fetchone()
        return dict(row) if row else None

    def list(self):
        with self.connection() as db:
            return [dict(r) for r in db.execute('SELECT * FROM videos ORDER BY position')]

    def update(self, video_id, **values):
        allowed={'state','phase','phase_progress','progress','detail','error','resource_state','detected_source_language'}
        if not set(values)<=allowed:
            raise ValueError('Invalid update fields')
        values['updated']=time.time()
        with self.connection() as db:
            db.execute('UPDATE videos SET '+','.join(k+'=?' for k in values)+' WHERE id=?',(*values.values(),video_id))

    def claim(self,editor_only=False,*,actions=None,exclude_actions=()):
        clauses=["state='queued'"];parameters=[]
        if editor_only:clauses.append("workflow_action!='full'")
        if actions is not None:
            if not actions:return None
            clauses.append('workflow_action IN ('+','.join('?' for _ in actions)+')');parameters.extend(actions)
        if exclude_actions:
            clauses.append('workflow_action NOT IN ('+','.join('?' for _ in exclude_actions)+')');parameters.extend(exclude_actions)
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM videos WHERE '+' AND '.join(clauses)+' ORDER BY position LIMIT 1',parameters).fetchone()
            if row:
                db.execute("UPDATE videos SET state='running',resource_state='local',phase='준비',phase_progress=0,progress=0,error=NULL,updated=? WHERE id=?",(time.time(),row['id']))
        return self.get(row['id']) if row else None

    def schedule_editor(self,video_id,action,index=None):
        if action not in {'analyze','tts','render','mix','watch'}:raise ValueError('Invalid editor action')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT state FROM videos WHERE id=?',(video_id,)).fetchone()
            if row is None:raise KeyError(video_id)
            if row['state'] in {'running','queued','deleting'}:raise RuntimeError('This job is already running.')
            db.execute("UPDATE videos SET state='queued',workflow_action=?,editor_index=?,phase='대기',phase_progress=0,progress=0,detail='',error=NULL,updated=? WHERE id=?",
                (action,index,time.time(),video_id))

    def delete(self,video_id):
        if str(uuid.UUID(video_id))!=video_id:raise ValueError('Invalid video ID')
        root=self.root.resolve()
        folders=[root/'work/videos'/video_id,root/'outputs'/video_id]
        log=root/'logs'/f'{video_id}.log'
        # Resolve every target before removing anything; never follow an ID directory link.
        for target in [*folders,log]:
            if target.resolve()!=target or not target.is_relative_to(root):
                raise RuntimeError('Deletion path is outside the job directory.')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT state FROM videos WHERE id=?',(video_id,)).fetchone()
            if row is None:raise KeyError(video_id)
            if row['state']=='running':raise RuntimeError('Stop the video processing job first.')
            db.execute("UPDATE videos SET state='deleting',phase='삭제 중',updated=? WHERE id=?",(time.time(),video_id))
        try:
            for folder in folders:
                if folder.resolve()!=folder:raise RuntimeError('Deletion path has changed.')
                if folder.exists():shutil.rmtree(folder)
            log.unlink(missing_ok=True)
        except (OSError,RuntimeError) as error:
            self.update(video_id,state='error',phase='삭제 실패',error='파일을 삭제하지 못했습니다. 사용 중인 파일을 닫고 다시 삭제해 주세요.')
            raise RuntimeError('Failed to delete files. Close files in use and retry deletion.') from error
        with self.connection() as db:
            db.execute('DELETE FROM videos WHERE id=?',(video_id,))

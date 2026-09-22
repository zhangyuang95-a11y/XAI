"""Deployment configuration; secrets never enter public release metadata."""
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class Settings:
    database: str
    persistent: bool = False
    admin_token: str = ''
    origin: str = 'http://127.0.0.1:8010'
    mode: str = 'preview'
    llm_base_url: str = ''
    llm_model: str = ''
    llm_api_key: str = ''
    verified: bool = False
    prolific_study_id: str = ''
    prolific_completion_code: str = ''
    prolific_places: int = 12
    prolific_launch_confirmed: bool = False

    @property
    def llm_configured(self):
        return bool(self.llm_base_url and self.llm_model)

    @property
    def ready(self):
        return self.mode=='pilot' and self.verified and self.persistent and self.llm_configured and bool(self.admin_token)

    @classmethod
    def from_env(cls):
        database = os.environ.get('POLICYLENS_DATABASE_URL') or os.environ.get('DATABASE_URL')
        database = database or str(Path('output/study_v3/study.sqlite3').resolve())
        postgres = database.startswith(('postgres://', 'postgresql://'))
        persistent = postgres or os.environ.get('POLICYLENS_STORAGE_MODE') == 'persistent'
        if persistent and not postgres and str(Path(database).resolve()).startswith(('/tmp/', '/private/tmp/')):
            raise ValueError('persistent_database_cannot_be_in_tmp')
        return cls(database=database, persistent=persistent,
            admin_token=os.environ.get('POLICYLENS_ADMIN_TOKEN',''),
            origin=os.environ.get('POLICYLENS_PUBLIC_ORIGIN','http://127.0.0.1:8010').rstrip('/'),
            mode=os.environ.get('POLICYLENS_MODE','pilot'),
            llm_base_url=os.environ.get('POLICYLENS_LLM_BASE_URL','').rstrip('/'),
            llm_model=os.environ.get('POLICYLENS_LLM_MODEL',''),
            llm_api_key=os.environ.get('POLICYLENS_LLM_API_KEY',''),
            verified=os.environ.get('POLICYLENS_STUDY_VERIFIED')=='1',
            prolific_study_id=os.environ.get('POLICYLENS_PROLIFIC_STUDY_ID',''),
            prolific_completion_code=os.environ.get('POLICYLENS_PROLIFIC_COMPLETION_CODE',''),
            prolific_places=int(os.environ.get('POLICYLENS_PROLIFIC_PLACES','12')),
            prolific_launch_confirmed=os.environ.get('POLICYLENS_PROLIFIC_LAUNCH_CONFIRMED')=='1')

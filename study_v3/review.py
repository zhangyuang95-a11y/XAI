"""Scoped invitations to the real study flow, excluded from paid recruitment."""
import base64
import hashlib
import hmac
import json
import secrets
import time

from . import RELEASE_ID
from .prolific import CONSENT_VERSION
from .registry import MODULES
from .store import StudyError, uid


def mint(settings, domain):
    if domain not in MODULES:
        raise StudyError('unknown_domain')
    if not settings.admin_token:
        raise StudyError('researcher_access_required', 403)
    claims = dict(domain=domain, release=RELEASE_ID, expires=int(time.time()) + 30*86400,
                  nonce=secrets.token_hex(8), limit=6)
    data = base64.urlsafe_b64encode(json.dumps(claims, separators=(',', ':')).encode()).decode().rstrip('=')
    signature = hmac.new(settings.admin_token.encode(), ('review:'+data).encode(), hashlib.sha256).hexdigest()
    return {'url': settings.origin+'/review/'+domain+'/#invite='+data+'.'+signature,
            'domain': domain, 'group': 'A', 'expires': claims['expires'], 'maximum_sessions': claims['limit']}


def validate(settings, domain, invitation):
    try:
        if not isinstance(invitation, str) or len(invitation)>2048 or not settings.admin_token:
            raise ValueError()
        data, signature = invitation.split('.')
        expected = hmac.new(settings.admin_token.encode(), ('review:'+data).encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError()
        claims = json.loads(base64.urlsafe_b64decode(data+'='*(-len(data)%4)))
        if claims['domain'] != domain or domain not in MODULES or claims['expires'] <= time.time():
            raise ValueError()
        if claims['release'] != RELEASE_ID:
            raise StudyError('review_release_changed', 409)
        return claims
    except (ValueError, KeyError, TypeError):
        raise StudyError('invalid_review_invitation', 403) from None


def enrol(store, domain, payload, token=None):
    claims = validate(store.settings, domain, payload.get('invitation'))
    if payload.get('consent') is not True or payload.get('age_21') is not True:
        raise StudyError('consent_required')
    if payload.get('consent_version') != CONSENT_VERSION:
        raise StudyError('consent_version_changed', 409)
    if not store.ready:
        raise StudyError('study_not_ready', 503)
    prefix = 'review-'+claims['nonce']+'-'
    with store.db.transaction() as db:
        participant = None
        if token:
            try:
                participant = store._participant(db, token)
            except StudyError:
                pass
        if participant and participant['id'].startswith(prefix):
            instance = db.one('SELECT * FROM pl3_instances WHERE participant_id=? AND domain=? AND release_id=?',
                              (participant['id'], domain, RELEASE_ID))
            if instance:
                return token, store._view(db, store._instance(db, token, instance['id']))
        count = db.one('SELECT COUNT(*) AS n FROM pl3_instances WHERE participant_id LIKE ?', (prefix+'%',))['n']
        if count >= claims['limit']:
            raise StudyError('review_invitation_full', 409)
        # Use the identical settings, engine, tutorials, rewards and explainer;
        # only assignment and accounting differ from a paid participant.
        return store.create(dict(participant_id=prefix+uid(), domain=domain, group='A',
                                 mode='preview', language='en', consent=True),
                            admin=True, _db=db)


def authorize(store, domain, token, instance_id=None):
    with store.db.transaction(read_only=True) as db:
        participant = store._participant(db, token)
        if not participant['id'].startswith('review-'):
            raise StudyError('review_session_required', 403)
        if instance_id:
            instance = store._instance(db, token, instance_id)
        else:
            instance = db.one('SELECT * FROM pl3_instances WHERE participant_id=? AND domain=? AND release_id=?',
                              (participant['id'], domain, RELEASE_ID))
        if not instance or instance['mode']!='preview' or instance['domain']!=domain or instance['group_code']!='A':
            raise StudyError('review_session_required', 403)
        return instance['id']

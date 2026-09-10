"""Read-only participant projection of a completed metadata-backed question bank.

The workflow reader verifies the saved full generation and independent replay.
This view cannot construct an environment, query an Actor, or repeat either run.
Content qualification stays separate from model, explanation and study admission.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from backend.training import warehouse_family_question_pool as question_pool
from backend.training.warehouse_native_common import ROOT, digest, file_hash
from ui import warehouse_alignment_metadata_bank as bank_api
from backend import warehouse_alignment_runtime as runtime_api
from ui.warehouse_family_bank_view import _items, _hash
from ui.warehouse_public_history_bank import PublicHistoryQuestionBank as _PureMethods

VERSION = 'warehouse-alignment-frozen-metadata-bank-view.v1'


def _same(left, right, reason):
    if digest(left) != digest(right):
        raise ValueError(reason)


def _registered(bank, root, anchor, fixture):
    """Check saved physical identities; do not restore any excluded scene."""
    registered = question_pool.read_registered(root,
        expected_manifest_sha256=anchor, allow_test_fixture=fixture)
    binding = {
        'version': registered['version'], 'manifest_sha256': anchor,
        'pool_sha256': registered['pool_sha256'],
        'source_scenario_manifest_sha256': registered['source_scenario_manifest_sha256'],
        'exclusion_binding': registered['exclusion_binding'],
        'sources_sha256': digest(registered['sources']),
        'artifacts': {name: {'sha256': sha256(raw).hexdigest(), 'size': len(raw)}
                      for name, raw in registered['blobs'].items()},
    }
    _same(bank['question_pool_binding'], binding, 'Saved question pool or complete exclusions differ')
    _same(bank['pool_scenes'], registered['scenes'], 'Question source scenes differ')
    _same(bank['excluded_fingerprints_sha256'], digest(sorted(registered['excluded_fingerprints'])),
          'Complete registered exclusion set differs')
    scenes = registered['scenes']
    if (registered['test_fixture'] is not fixture
            or registered['qualification_granted'] is not False
            or len(scenes) != (4 if fixture else 24)):
        raise ValueError('Question registration scope differs')
    # read_registered already recomputes the physical fingerprints from saved
    # public contents. These sets also bind the participant view to that result.
    fingerprints = [s['fingerprint'] for s in scenes]
    if (len({s['id'] for s in scenes}) != len(scenes)
            or len(set(fingerprints)) != len(scenes)
            or set(fingerprints) & set(registered['excluded_fingerprints'])):
        raise ValueError('Question scenes overlap their registered sources')
    return registered, {scene['id']: scene for scene in scenes}


class FrozenAlignmentMetadataBank:
    """An immutable in-memory bank with externally anchored saved evidence."""
    def __init__(self, workflow_root, *, expected_prepared_sha256,
                 expected_replay_receipt_sha256, expected_runtime_signature,
                 expected_actor_sha256, expected_protocol_sha256,
                 expected_scenario_manifest_sha256, expected_pool_manifest_sha256,
                 allow_test_fixture=False):
        anchors = dict(prepared=expected_prepared_sha256,
            replay_receipt=expected_replay_receipt_sha256,
            runtime_signature=expected_runtime_signature, actor=expected_actor_sha256,
            protocol=expected_protocol_sha256, scenarios=expected_scenario_manifest_sha256,
            question_pool=expected_pool_manifest_sha256)
        for name, value in anchors.items():
            _hash(value, name)
        if type(allow_test_fixture) is not bool:
            raise ValueError('Fixture scope must be an explicit boolean')
        # Importing a reader does not execute generation or load an Actor.
        from backend.training import warehouse_alignment_metadata_bank_run as driver
        root = driver._absolute(workflow_root)
        receipt_raw = driver._read(root/'completion_receipt.json')
        if sha256(receipt_raw).hexdigest() != expected_replay_receipt_sha256:
            raise ValueError('External replay receipt byte anchor differs')
        result = driver.read_completed(root, expected_prepared_sha256=expected_prepared_sha256,
                                       allow_test_fixture=allow_test_fixture)
        _same(result['replay_receipt'], json.loads(receipt_raw), 'Replay receipt changed during loading')
        if result['replay_receipt_sha256'] != expected_replay_receipt_sha256:
            raise ValueError('Completed replay receipt anchor differs')
        bank, receipt = result['bank_private_data'], result['replay_receipt']
        if (set(bank) != bank_api._FIELDS or bank['version'] != bank_api.VERSION
                or bank['status'] != 'candidate' or bank['scope'] != bank_api.SCOPE
                or bank['formal_ready'] is not False or bank['release_ready'] is not False
                or bank['test_fixture'] is not allow_test_fixture
                or bank['runtime_family'] != runtime_api.FAMILY
                or bank['runtime_signature'] != expected_runtime_signature
                or bank['actor_sha256'] != expected_actor_sha256
                or bank['protocol_sha256'] != expected_protocol_sha256
                or bank['pool_namespace'] != bank_api.POOL_NAMESPACE
                or bank['counterfactual_filter'] != bank_api.FILTER
                or receipt['runtime']['runtime_signature'] != expected_runtime_signature
                or receipt['runtime']['actor_sha256'] != expected_actor_sha256
                or result['report']['independent_replay_completed'] is not True):
            raise ValueError('Selected Actor, runtime or completed bank scope differs')
        registered, scenes = _registered(bank, root/'question_pool',
            expected_pool_manifest_sha256, allow_test_fixture)
        if registered['source_scenario_manifest_sha256'] != expected_scenario_manifest_sha256:
            raise ValueError('Question bank descends from a different original scenario manifest')
        if not allow_test_fixture and (registered['configuration']['horizon'] != 120
                or bank['trajectory_steps'] != 24 or bank['minimum_frame'] != 1):
            raise ValueError('Frozen production question interval differs')
        self._checks = _items(bank, scenes)
        self._bank, self._items, self._receipt = deepcopy(bank), deepcopy(bank['items']), deepcopy(receipt)
        self._pool_binding = deepcopy(bank['question_pool_binding'])
        self.test_fixture = allow_test_fixture
        self.actor_sha256, self.runtime_signature = expected_actor_sha256, expected_runtime_signature
        self.protocol_sha256 = expected_protocol_sha256
        self.pool_manifest_sha256 = expected_pool_manifest_sha256
        self.bank_sha256 = result['bank_binding']['sha256']
        self.replay_receipt_sha256 = expected_replay_receipt_sha256
        prepared_raw = driver._read(root/'prepared.json')
        if sha256(prepared_raw).hexdigest() != expected_prepared_sha256:
            raise ValueError('Prepared evidence changed during loading')
        self._sources = {**json.loads(prepared_raw)['identity']['sources'],
            str(Path(__file__).relative_to(ROOT)): file_hash(__file__),
            'ui/warehouse_family_bank_view.py': file_hash(ROOT/'ui/warehouse_family_bank_view.py')}
        self.signature = digest(dict(version=VERSION, anchors=anchors,
            bank_sha256=self.bank_sha256, sources=self._sources))
        self._root, self._anchors = root, deepcopy(anchors)
        self._bound_files = {name: driver._binding(driver._read(root/name)) for name in
            ('prepared.json', 'completion_receipt.json', 'bank.private.json', 'head.json',
             'generation_report.json', 'verification_report.json')}
        for name, binding in result['prepared']['identity']['inputs'].items():
            self._bound_files['inputs/'+name] = deepcopy(binding)
        for name, binding in self._pool_binding['artifacts'].items():
            self._bound_files['question_pool/'+name] = deepcopy(binding)
        self._memory_sha256 = digest(self._memory_state())
        self.verify_binding()

    def _memory_state(self):
        return dict(bank=self._bank, items=self._items, receipt=self._receipt, checks=self._checks,
            pool=self._pool_binding, anchors=self._anchors, files=self._bound_files, root=str(self._root),
            identity=[self.test_fixture, self.actor_sha256, self.runtime_signature, self.protocol_sha256,
                self.pool_manifest_sha256, self.bank_sha256, self.replay_receipt_sha256])

    def verify_binding(self):
        """Zero-forward cached-content/source verification; return the view signature.

        The complete raw operation journal was checked at construction. This
        check anchors the resulting immutable receipt, inputs/reports and private
        content; it does not repeat the journal traversal or physical replay.
        """
        from backend.training import warehouse_alignment_metadata_bank_run as driver
        if digest(self._memory_state()) != self._memory_sha256:
            raise ValueError('Frozen alignment bank content or identity changed')
        for name, binding in self._bound_files.items():
            _same(driver._binding(driver._read(self._root/name)), binding, 'Frozen bank evidence bytes changed')
        driver._current(self._sources)
        expected = digest(dict(version=VERSION, anchors=self._anchors,
            bank_sha256=self.bank_sha256, sources=self._sources))
        if self.signature != expected:
            raise ValueError('Frozen alignment bank signature changed')
        return self.signature


    @property
    def content_eligible(self): return True
    @property
    def eligible(self): return False
    @property
    def participant_enabled(self): return False
    @property
    def formal_ready(self): return False
    @property
    def release_ready(self): return False
    @property
    def checks(self): return deepcopy(self._checks)
    @property
    def bank(self): return deepcopy(self._bank)
    @property
    def items(self): return deepcopy(self._items)
    @property
    def replay_receipt(self): return deepcopy(self._receipt)
    @property
    def question_pool_binding(self): return deepcopy(self._pool_binding)
    @property
    def sources(self): return deepcopy(self._sources)

    def public_items(self): return _PureMethods.public_items(self)
    def summary(self):
        return {**_PureMethods.summary(self), 'version': VERSION, 'eligible': False,
            'participant_enabled': False, 'independent_replay_previously_verified': True,
            'physics_replay_on_load': False, 'runtime_family': self._bank['runtime_family'],
            'bank_sha256': self.bank_sha256, 'replay_receipt_sha256': self.replay_receipt_sha256}
    def grade(self, answers):
        if type(answers) is not dict:
            raise ValueError('Prediction answers must be an object')
        return _PureMethods.grade(self, answers)

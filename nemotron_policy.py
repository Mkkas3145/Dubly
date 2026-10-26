"""Lightweight identity shared by the Nemotron worker and cache adapter."""
from pathlib import Path
import hashlib
MODEL_REVISION='f667ed73aee57d40cc39428eb768b4fd87a0a29e'
EMBEDDING_SHA='287365f6f485b19e65e5176554f8f7123bfa8d85185f3d2c040eab51acec9868'
VERSION=4
PROFILE='animation_native_reference_consensus_v1'
WINDOW_POLICY={'window_seconds':60.,'overlap_seconds':20.,'minimum_adaptive_seconds':10.,
    'split_when_local_slots_exhausted':True,'embedding_min_seconds':1.,
    'activity_threshold':.5,
    'split_at_local_speaker_count':7,
    'embedding_pure_probability':.9,'embedding_max_samples':8,'embedding_sample_seconds':2.,
    'embedding_sampling':'contiguous_exclusive_speech_only_no_spliced_frames'}

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def identity_for(source, root):
    from nemotron_linking import POLICY as LINK_POLICY
    model_path=Path(root)/'models/nemotron-3-diarization'
    if (model_path/'revision.txt').read_text().strip()!=MODEL_REVISION:
        raise RuntimeError('Nemotron model revision mismatch')
    from nemotron_context_linking import POLICY as REIDENTIFICATION_POLICY
    from nemotron_local_consensus import POLICY as LOCAL_CONSENSUS_POLICY
    return {'version':VERSION,'profile':PROFILE,'source_sha256':digest(source),'model':'nvidia/Nemotron-3-Diarization',
        'revision':MODEL_REVISION,'external_embedding_model':None,
        'window_policy':WINDOW_POLICY,'link_policy':LINK_POLICY,'reidentification':REIDENTIFICATION_POLICY,'local_consensus':LOCAL_CONSENSUS_POLICY}

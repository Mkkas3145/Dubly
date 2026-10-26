"""Production Nemotron-only cache identity; no external encoder dependency."""
from nemotron_policy import identity_for as native_identity
from nemotron_packed_linking import POLICY

def identity_for(source,root):
    return {**native_identity(source,root),'packed_native_linking':POLICY,
        'global_refinement':'single_shared_bank_pass_then_recheck_only_after_local_identity_repair',
        'external_embedding_model':None}

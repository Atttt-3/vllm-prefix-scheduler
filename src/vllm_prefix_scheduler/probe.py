"""Length-only lookup for the validated single full-attention native cache.

The read-only call chain was inspected in vLLM 0.32.0 at commit
5df63a7203cdb9328d824a27b682e77c7b9928be: KVCacheManager's lookup predicate,
UnitaryKVCacheCoordinator.find_longest_cache_hit, FullAttentionManager, and
BlockPool.get_cached_block. Upstream attribution: Apache-2.0, vLLM contributors.

Unlike KVCacheManager.get_computed_blocks, this path does not emit cached-block
events for full reports. The scheduler validates the supported cache shape at
construction. No returned blocks escape this function or replace native lookup
and allocation later in the scheduling step.
"""


def reusable_prefix_tokens(manager, request) -> int:
    """Read current reusable length; preserve native skip-read and last-token rules."""
    if not manager.prefix_cache_lookup_enabled(request):
        return 0
    _, hit_tokens, _ = manager.coordinator.find_longest_cache_hit(
        request.block_hashes, request.num_tokens - 1
    )
    return hit_tokens

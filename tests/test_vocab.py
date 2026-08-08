from functools import partial

import jax
import jax.numpy as jnp
import pytest

from mapox.vocab import FrozenVocabularyError, Vocabulary


# --- construction / serialization ---


def test_empty():
    v = Vocabulary()
    assert len(v) == 0
    assert v.symbols == ()
    assert list(v) == []


def test_init_assigns_ids_in_order():
    v = Vocabulary(["a", "b", "c"])
    assert [v.id(s) for s in ("a", "b", "c")] == [0, 1, 2]
    assert v.symbols == ("a", "b", "c")


def test_init_ignores_duplicates():
    v = Vocabulary(["a", "b", "a"])
    assert v.symbols == ("a", "b")


def test_round_trip():
    v = Vocabulary(["tile/wall", "tile/empty", "move/up"])
    assert Vocabulary(list(v)) == v


# --- add ---


def test_add_returns_sequential_ids():
    v = Vocabulary()
    assert v.add("a") == 0
    assert v.add("b") == 1
    assert v.symbols == ("a", "b")


def test_add_is_idempotent():
    v = Vocabulary(["a", "b"])
    assert v.add("a") == 0
    assert len(v) == 2


def test_add_known_symbol_allowed_when_frozen():
    v = Vocabulary(["a"]).freeze()
    assert v.add("a") == 0


def test_add_unknown_symbol_raises_when_frozen():
    v = Vocabulary(["a"]).freeze()
    with pytest.raises(FrozenVocabularyError):
        v.add("b")


# --- lookups ---


def test_id_raises_keyerror_naming_symbol():
    with pytest.raises(KeyError, match="tile/wal"):
        Vocabulary(["tile/wall"]).id("tile/wal")


def test_get_returns_none_for_unknown():
    v = Vocabulary(["a"])
    assert v.get("a") == 0
    assert v.get("b") is None


def test_contains():
    v = Vocabulary(["a"])
    assert "a" in v
    assert "b" not in v


# --- extend ---


def test_extend_appends_unseen_in_order():
    v = Vocabulary(["a", "b"])
    v.extend(["b", "c", "d"])
    assert v.symbols == ("a", "b", "c", "d")


def test_extend_on_frozen_with_all_known_is_noop():
    v = Vocabulary(["a", "b"]).freeze()
    v.extend(["a", "b"])
    assert v.symbols == ("a", "b")


def test_extend_on_frozen_with_new_symbol_raises():
    v = Vocabulary(["a"]).freeze()
    with pytest.raises(FrozenVocabularyError):
        v.extend(["a", "b"])


# --- freeze ---


def test_unfrozen_by_default():
    assert not Vocabulary().frozen


def test_freeze_returns_self_and_is_idempotent():
    v = Vocabulary(["a"])
    assert v.freeze() is v
    assert v.frozen
    v.freeze()
    assert v.frozen
    assert v.id("a") == 0  # lookups stay allowed


# --- add_block ---


def test_add_block_fresh_appends_contiguously():
    v = Vocabulary(["x"])
    block = v.add_block(["a", "b", "c"])
    assert block == range(1, 4)
    assert [v.id(s) for s in ("a", "b", "c")] == [1, 2, 3]


def test_add_block_single_symbol():
    v = Vocabulary()
    assert v.add_block(["a"]) == range(0, 1)


def test_add_block_is_idempotent():
    v = Vocabulary()
    first = v.add_block(["a", "b"])
    assert v.add_block(["a", "b"]) == first
    assert len(v) == 2


def test_add_block_partial_overlap_raises():
    v = Vocabulary(["a"])
    with pytest.raises(ValueError):
        v.add_block(["a", "b"])


def test_add_block_non_contiguous_raises():
    v = Vocabulary(["a", "x", "b"])
    with pytest.raises(ValueError):
        v.add_block(["a", "b"])


def test_add_block_wrong_order_raises():
    v = Vocabulary(["a", "b"])
    with pytest.raises(ValueError):
        v.add_block(["b", "a"])


def test_add_block_rejects_duplicate_symbols():
    with pytest.raises(ValueError):
        Vocabulary().add_block(["a", "a"])


def test_add_block_rejects_empty():
    with pytest.raises(ValueError):
        Vocabulary().add_block([])


def test_add_block_fresh_on_frozen_raises():
    v = Vocabulary(["a"]).freeze()
    with pytest.raises(FrozenVocabularyError):
        v.add_block(["b", "c"])


def test_add_block_existing_on_frozen_is_allowed():
    # An env re-resolving its block against a frozen (e.g. loaded) vocab.
    v = Vocabulary(["a", "b"]).freeze()
    assert v.add_block(["a", "b"]) == range(0, 2)


# --- equality ---


def test_eq_by_symbol_order():
    assert Vocabulary(["a", "b"]) == Vocabulary(["a", "b"])
    assert Vocabulary(["a", "b"]) != Vocabulary(["b", "a"])


def test_eq_ignores_frozen():
    assert Vocabulary(["a"]).freeze() == Vocabulary(["a"])


# --- hashing / jit static args ---


def test_unfrozen_vocab_is_unhashable():
    with pytest.raises(TypeError, match="freeze"):
        hash(Vocabulary(["a"]))


def test_frozen_vocabs_hash_by_content():
    a = Vocabulary(["x", "y"]).freeze()
    b = Vocabulary(["x", "y"]).freeze()
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_frozen_vocab_as_jit_static_arg():
    @partial(jax.jit, static_argnums=0)
    def f(vocab, x):
        return x + len(vocab)

    v = Vocabulary(["a", "b"]).freeze()
    assert f(v, jnp.int32(1)) == 3
    # An equal-but-distinct frozen vocab hits the same cache entry.
    assert f(Vocabulary(["a", "b"]).freeze(), jnp.int32(2)) == 4


# --- lut_to ---


def test_lut_to_maps_into_target_ids():
    local = Vocabulary(["wall", "food"])
    global_ = Vocabulary(["empty", "food", "wall"])
    lut = local.lut_to(global_)
    assert lut.tolist() == [2, 1]
    assert lut.shape == (len(local),)


def test_lut_to_missing_symbol_raises_naming_it():
    a = Vocabulary(["wall", "lava"])
    b = Vocabulary(["wall"])
    with pytest.raises(KeyError, match="lava"):
        a.lut_to(b)


def test_lut_to_default_fills_missing():
    # Action direction: global → local, unsupported actions hit the fallback.
    global_ = Vocabulary(["up", "down", "dig"])
    local = Vocabulary(["down", "up"])
    print(local.id("up"))
    print(local.id("down"))
    lut = global_.lut_to(local, default=1)
    assert lut.tolist() == [1, 0, 1]


def test_lut_to_dtypes():
    v = Vocabulary(["a"])
    assert v.lut_to(v).dtype == jnp.uint16
    assert v.lut_to(v, dtype=jnp.int32).dtype == jnp.int32

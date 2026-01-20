"""
Tests for the Continuum Memory System.

Tests the multi-scale temporal memory system implementing the HOPE architecture
from Nested Learning (Behrouz et al., NeurIPS 2025).
"""

import sys
import os
import importlib.util

import pytest

# Load module directly from file to avoid circular imports through package __init__.py
def _load_module_from_file(module_name: str, file_path: str):
    """Load a module directly from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Get the project root
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_continuum_memory_path = os.path.join(_project_root, 'core', 'memory', 'continuum_memory.py')

# Load the module directly
_cm = _load_module_from_file('continuum_memory_direct', _continuum_memory_path)

# Import classes from loaded module
MemoryBankConfig = _cm.MemoryBankConfig
MemoryEntry = _cm.MemoryEntry
MemoryBank = _cm.MemoryBank
CrossTemporalAttention = _cm.CrossTemporalAttention
ContinuumMemoryConfig = _cm.ContinuumMemoryConfig
ContinuumMemorySystem = _cm.ContinuumMemorySystem
create_continuum_memory = _cm.create_continuum_memory
get_global_continuum_memory = _cm.get_global_continuum_memory


class TestMemoryBankConfig:
    """Tests for MemoryBankConfig."""

    def test_create_config(self):
        """Test creating a memory bank configuration."""
        config = MemoryBankConfig(
            name='test_bank',
            capacity=100,
            update_freq=10,
            decay_rate=0.1
        )
        assert config.name == 'test_bank'
        assert config.capacity == 100
        assert config.update_freq == 10
        assert config.decay_rate == 0.1

    def test_default_values(self):
        """Test default values in config."""
        config = MemoryBankConfig(
            name='test',
            capacity=50,
            update_freq=5,
            decay_rate=0.05
        )
        assert config.importance_threshold == 0.5
        assert config.enable_consolidation is True


class TestMemoryEntry:
    """Tests for MemoryEntry."""

    def test_create_entry(self):
        """Test creating a memory entry."""
        entry = MemoryEntry(
            content="test content",
            importance=0.8,
            timestamp=100
        )
        assert entry.content == "test content"
        assert entry.importance == 0.8
        assert entry.timestamp == 100
        assert entry.access_count == 0

    def test_update_access(self):
        """Test updating access statistics."""
        entry = MemoryEntry(content="test", importance=0.5, timestamp=0)
        entry.update_access(50)
        assert entry.access_count == 1
        assert entry.last_access == 50

        entry.update_access(100)
        assert entry.access_count == 2
        assert entry.last_access == 100

    def test_calculate_relevance(self):
        """Test relevance calculation."""
        entry = MemoryEntry(content="test", importance=0.8, timestamp=0)
        relevance = entry.calculate_relevance(current_step=10)
        assert 0 <= relevance <= 1

        # Higher importance should lead to higher relevance
        entry_high = MemoryEntry(content="high", importance=0.9, timestamp=0)
        entry_low = MemoryEntry(content="low", importance=0.3, timestamp=0)

        rel_high = entry_high.calculate_relevance(10)
        rel_low = entry_low.calculate_relevance(10)
        assert rel_high > rel_low

    def test_relevance_with_access(self):
        """Test that access count affects relevance."""
        entry = MemoryEntry(content="test", importance=0.5, timestamp=0)
        rel_before = entry.calculate_relevance(10)

        # Simulate multiple accesses
        for _ in range(5):
            entry.update_access(10)

        rel_after = entry.calculate_relevance(10)
        assert rel_after >= rel_before


class TestMemoryBank:
    """Tests for MemoryBank."""

    def test_create_bank(self):
        """Test creating a memory bank."""
        config = MemoryBankConfig(
            name='test_bank',
            capacity=10,
            update_freq=1,
            decay_rate=0.1,
            importance_threshold=0.0
        )
        bank = MemoryBank(config)
        assert len(bank.memories) == 0
        assert bank.total_writes == 0
        assert bank.total_reads == 0

    def test_should_update(self):
        """Test update frequency check."""
        config = MemoryBankConfig(
            name='test',
            capacity=10,
            update_freq=5,
            decay_rate=0.1
        )
        bank = MemoryBank(config)

        assert bank.should_update(0) is True
        assert bank.should_update(1) is False
        assert bank.should_update(5) is True
        assert bank.should_update(10) is True

    def test_write_memory(self):
        """Test writing memory to bank."""
        config = MemoryBankConfig(
            name='test',
            capacity=10,
            update_freq=1,
            decay_rate=0.1,
            importance_threshold=0.0
        )
        bank = MemoryBank(config)

        entry = MemoryEntry(content="test", importance=0.8, timestamp=1)
        success = bank.write(entry, current_step=1)

        assert success is True
        assert len(bank.memories) == 1
        assert bank.total_writes == 1

    def test_write_rejected_by_frequency(self):
        """Test that writes are rejected when frequency doesn't match."""
        config = MemoryBankConfig(
            name='test',
            capacity=10,
            update_freq=10,
            decay_rate=0.1,
            importance_threshold=0.0
        )
        bank = MemoryBank(config)

        entry = MemoryEntry(content="test", importance=0.8, timestamp=5)
        success = bank.write(entry, current_step=5)  # Not multiple of 10

        assert success is False
        assert len(bank.memories) == 0

    def test_write_rejected_by_importance(self):
        """Test that low importance entries are rejected."""
        config = MemoryBankConfig(
            name='test',
            capacity=10,
            update_freq=1,
            decay_rate=0.1,
            importance_threshold=0.7
        )
        bank = MemoryBank(config)

        entry = MemoryEntry(content="test", importance=0.5, timestamp=1)
        success = bank.write(entry, current_step=1)

        assert success is False
        assert len(bank.memories) == 0

    def test_read_memory(self):
        """Test reading memories from bank."""
        config = MemoryBankConfig(
            name='test',
            capacity=10,
            update_freq=1,
            decay_rate=0.0,
            importance_threshold=0.0
        )
        bank = MemoryBank(config)

        # Write some memories
        for i in range(5):
            entry = MemoryEntry(content=f"content_{i}", importance=i * 0.2, timestamp=i)
            bank.write(entry, current_step=i)

        # Read top 3
        memories = bank.read(current_step=5, top_k=3)
        assert len(memories) == 3
        assert bank.total_reads == 1

    def test_read_empty_bank(self):
        """Test reading from empty bank."""
        config = MemoryBankConfig(
            name='test',
            capacity=10,
            update_freq=1,
            decay_rate=0.1,
            importance_threshold=0.0
        )
        bank = MemoryBank(config)

        memories = bank.read(current_step=1, top_k=5)
        assert len(memories) == 0

    def test_apply_decay(self):
        """Test memory decay."""
        config = MemoryBankConfig(
            name='test',
            capacity=10,
            update_freq=1,
            decay_rate=0.5,
            importance_threshold=0.3
        )
        bank = MemoryBank(config)

        entry = MemoryEntry(content="test", importance=0.6, timestamp=0)
        bank.write(entry, current_step=0)

        # Apply decay
        bank.apply_decay(current_step=1)

        # Check that importance decreased but memory still exists
        assert len(bank.memories) == 1
        assert bank.memories[0].importance < 0.6

    def test_decay_removes_low_importance(self):
        """Test that decay removes memories below threshold."""
        config = MemoryBankConfig(
            name='test',
            capacity=10,
            update_freq=1,
            decay_rate=0.9,
            importance_threshold=0.5
        )
        bank = MemoryBank(config)

        entry = MemoryEntry(content="test", importance=0.55, timestamp=0)
        bank.write(entry, current_step=0)

        # Apply heavy decay
        bank.apply_decay(current_step=1)

        # Memory should be removed (0.55 * 0.1 = 0.055 < 0.5)
        assert len(bank.memories) == 0

    def test_capacity_limit(self):
        """Test that bank respects capacity limit."""
        config = MemoryBankConfig(
            name='test',
            capacity=3,
            update_freq=1,
            decay_rate=0.0,
            importance_threshold=0.0
        )
        bank = MemoryBank(config)

        # Write more than capacity
        for i in range(5):
            entry = MemoryEntry(content=f"content_{i}", importance=0.5, timestamp=i)
            bank.write(entry, current_step=i)

        # Should only keep capacity number of entries
        assert len(bank.memories) == 3

    def test_get_important_memories(self):
        """Test getting important memories for consolidation."""
        config = MemoryBankConfig(
            name='test',
            capacity=10,
            update_freq=1,
            decay_rate=0.0,
            importance_threshold=0.0
        )
        bank = MemoryBank(config)

        # Write memories with varying importance
        for i in range(5):
            entry = MemoryEntry(content=f"content_{i}", importance=i * 0.2 + 0.1, timestamp=i)
            bank.write(entry, current_step=i)

        # Get memories above 0.5 importance
        important = bank.get_important_memories(threshold=0.5, current_step=5)
        assert all(m.importance >= 0.5 for m in important)

    def test_consolidate(self):
        """Test consolidating memories from another bank."""
        config = MemoryBankConfig(
            name='test',
            capacity=10,
            update_freq=1,
            decay_rate=0.0,
            importance_threshold=0.0
        )
        bank = MemoryBank(config)

        # Create memories to consolidate
        memories = [
            MemoryEntry(content=f"content_{i}", importance=0.5, timestamp=i)
            for i in range(3)
        ]

        bank.consolidate(memories)

        assert len(bank.memories) == 3
        # Importance should be boosted (1.2x)
        for mem in bank.memories:
            assert mem.importance == 0.6

    def test_get_statistics(self):
        """Test getting bank statistics."""
        config = MemoryBankConfig(
            name='test_stats',
            capacity=10,
            update_freq=1,
            decay_rate=0.1,
            importance_threshold=0.0
        )
        bank = MemoryBank(config)

        # Write some memories
        for i in range(3):
            entry = MemoryEntry(content=f"content_{i}", importance=0.5, timestamp=i)
            bank.write(entry, current_step=i)

        stats = bank.get_statistics()

        assert stats['name'] == 'test_stats'
        assert stats['capacity'] == 10
        assert stats['current_size'] == 3
        assert stats['utilization'] == 0.3
        assert stats['total_writes'] == 3


class TestCrossTemporalAttention:
    """Tests for CrossTemporalAttention."""

    def test_create_attention(self):
        """Test creating cross-temporal attention."""
        attention = CrossTemporalAttention(temperature=1.0)
        assert attention.temperature == 1.0
        assert len(attention.attention_history) == 0

    def test_attend_empty(self):
        """Test attending with no memories."""
        attention = CrossTemporalAttention()
        result = attention.attend(query=None, memory_contents={})

        assert result['combined_memories'] == []
        assert result['attention_weights'] == {}

    def test_attend_with_memories(self):
        """Test attending with memories from multiple banks."""
        attention = CrossTemporalAttention()

        memory_contents = {
            'ultra_short': [
                MemoryEntry(content='recent1', importance=0.8, timestamp=10),
                MemoryEntry(content='recent2', importance=0.6, timestamp=9),
            ],
            'long': [
                MemoryEntry(content='old1', importance=0.9, timestamp=1),
            ]
        }

        result = attention.attend(query=None, memory_contents=memory_contents)

        assert len(result['combined_memories']) > 0
        assert 'ultra_short' in result['attention_weights']
        assert 'long' in result['attention_weights']

    def test_attend_with_custom_weights(self):
        """Test attending with custom bank weights."""
        attention = CrossTemporalAttention()

        memory_contents = {
            'ultra_short': [
                MemoryEntry(content='recent', importance=0.5, timestamp=10),
            ],
            'long': [
                MemoryEntry(content='old', importance=0.5, timestamp=1),
            ]
        }

        custom_weights = {
            'ultra_short': 0.9,
            'long': 0.1
        }

        result = attention.attend(
            query=None,
            memory_contents=memory_contents,
            bank_weights=custom_weights
        )

        assert result['attention_weights']['ultra_short'] == 0.9
        assert result['attention_weights']['long'] == 0.1


class TestContinuumMemoryConfig:
    """Tests for ContinuumMemoryConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ContinuumMemoryConfig()

        assert config.ultra_short_capacity == 16
        assert config.short_capacity == 256
        assert config.medium_capacity == 2048
        assert config.long_capacity == 8192

        assert config.ultra_short_freq == 1
        assert config.short_freq == 10
        assert config.medium_freq == 100
        assert config.long_freq == 1000

    def test_custom_config(self):
        """Test custom configuration."""
        config = ContinuumMemoryConfig(
            ultra_short_capacity=32,
            short_freq=5,
            consolidation_interval=500
        )

        assert config.ultra_short_capacity == 32
        assert config.short_freq == 5
        assert config.consolidation_interval == 500


class TestContinuumMemorySystem:
    """Tests for ContinuumMemorySystem."""

    def test_create_system(self):
        """Test creating memory system."""
        system = ContinuumMemorySystem()

        assert 'ultra_short' in system.memory_banks
        assert 'short' in system.memory_banks
        assert 'medium' in system.memory_banks
        assert 'long' in system.memory_banks
        assert system.current_step == 0

    def test_create_with_custom_config(self):
        """Test creating system with custom config."""
        config = ContinuumMemoryConfig(ultra_short_capacity=32)
        system = ContinuumMemorySystem(config)

        assert system.memory_banks['ultra_short'].config.capacity == 32

    def test_write_memory(self):
        """Test writing memory to system."""
        system = ContinuumMemorySystem()

        results = system.write_memory(
            content="test content",
            importance=0.8,
            metadata={'key': 'value'}
        )

        assert system.current_step == 1
        assert 'ultra_short' in results

    def test_write_memory_multiple_times(self):
        """Test writing multiple memories."""
        system = ContinuumMemorySystem()

        for i in range(20):
            system.write_memory(
                content=f"content_{i}",
                importance=0.5 + i * 0.02
            )

        assert system.current_step == 20
        # Ultra-short should have written every step
        assert system.memory_banks['ultra_short'].total_writes > 0

    def test_read_memory(self):
        """Test reading memory from system."""
        system = ContinuumMemorySystem()

        # Write some memories
        for i in range(10):
            system.write_memory(content=f"content_{i}", importance=0.6)

        result = system.read_memory(top_k=5)

        assert 'combined_memories' in result
        assert 'attention_weights' in result

    def test_consolidate_memory(self):
        """Test memory consolidation."""
        config = ContinuumMemoryConfig(
            consolidation_interval=10,
            consolidation_threshold=0.5
        )
        system = ContinuumMemorySystem(config)

        # Write many memories to trigger consolidation
        for i in range(15):
            system.write_memory(content=f"important_{i}", importance=0.8)

        # Consolidation should have been triggered
        assert system.last_consolidation_step > 0

    def test_get_statistics(self):
        """Test getting system statistics."""
        system = ContinuumMemorySystem()

        for i in range(5):
            system.write_memory(content=f"content_{i}", importance=0.5)

        stats = system.get_statistics()

        assert stats['current_step'] == 5
        assert 'total_capacity' in stats
        assert 'total_memories' in stats
        assert 'global_utilization' in stats
        assert 'banks' in stats

    def test_reset(self):
        """Test resetting the system."""
        system = ContinuumMemorySystem()

        # Write some memories
        for i in range(10):
            system.write_memory(content=f"content_{i}", importance=0.5)

        assert system.current_step > 0

        # Reset
        system.reset()

        assert system.current_step == 0
        assert system.last_consolidation_step == 0
        for bank in system.memory_banks.values():
            assert len(bank.memories) == 0


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_continuum_memory(self):
        """Test factory function."""
        system = create_continuum_memory()
        assert isinstance(system, ContinuumMemorySystem)

    def test_create_with_custom_config(self):
        """Test factory with custom config."""
        config = ContinuumMemoryConfig(ultra_short_capacity=64)
        system = create_continuum_memory(config)

        assert system.config.ultra_short_capacity == 64

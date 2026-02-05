#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Blockchain-like Audit Log for Training Data Traceability
===========================================================

Immutable audit trail system that creates a blockchain-like structure
for tracking all training data influences on model parameters.

Features:
- Cryptographic hashing for immutability
- Chain-of-custody for data provenance
- Tamper-evident audit trail
- Real-time verification
- Compliance-ready export formats
"""

import hashlib
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from Crypto.Hash import SHA256
from Crypto.Signature import pkcs1_15
from Crypto.PublicKey import RSA

logger = logging.getLogger(__name__)

@dataclass
class DataProvenanceHash:
    """Cryptographic hash for data provenance."""
    dataset_id: str
    data_hash: str
    timestamp: float
    size_bytes: int
    source_url: Optional[str] = None
    preprocessing_hash: Optional[str] = None
    
    def compute_hash(self) -> str:
        """Compute SHA-256 hash of all provenance data."""
        data_string = f"{self.dataset_id}:{self.data_hash}:{self.timestamp}:{self.size_bytes}"
        if self.source_url:
            data_string += f":{self.source_url}"
        if self.preprocessing_hash:
            data_string += f":{self.preprocessing_hash}"
        return hashlib.sha256(data_string.encode()).hexdigest()

@dataclass 
class ImmutableLogEntry:
    """Single immutable log entry with cryptographic integrity."""
    entry_id: str
    timestamp: float
    entry_type: str  # "dataset_load", "parameter_update", "gradient_step", etc.
    dataset_id: str
    affected_parameters: List[str]
    parameter_deltas: Dict[str, float]
    gradient_norms: Dict[str, float]
    data_provenance: DataProvenanceHash
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def compute_entry_hash(self) -> str:
        """Compute cryptographic hash of this entry."""
        entry_dict = {
            'entry_id': self.entry_id,
            'timestamp': self.timestamp,
            'entry_type': self.entry_type,
            'dataset_id': self.dataset_id,
            'affected_parameters': sorted(self.affected_parameters),
            'parameter_deltas': dict(sorted(self.parameter_deltas.items())),
            'gradient_norms': dict(sorted(self.gradient_norms.items())),
            'data_provenance_hash': self.data_provenance.compute_hash()
        }
        entry_json = json.dumps(entry_dict, sort_keys=True)
        return hashlib.sha256(entry_json.encode()).hexdigest()

@dataclass
class AuditBlock:
    """Blockchain-like block containing multiple audit entries."""
    block_id: str
    previous_block_hash: str
    timestamp: float
    entries: List[ImmutableLogEntry]
    merkle_root: str
    block_hash: str = ""
    nonce: int = 0
    
    def compute_merkle_root(self) -> str:
        """Compute Merkle tree root for all entries in block."""
        if not self.entries:
            return "0" * 64
        
        hashes = [entry.compute_entry_hash() for entry in self.entries]
        
        # Build Merkle tree
        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])  # Duplicate last hash if odd number
            
            new_hashes = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i + 1]
                new_hashes.append(hashlib.sha256(combined.encode()).hexdigest())
            hashes = new_hashes
        
        return hashes[0]
    
    def compute_block_hash(self) -> str:
        """Compute hash of entire block."""
        block_data = {
            'block_id': self.block_id,
            'previous_block_hash': self.previous_block_hash,
            'timestamp': self.timestamp,
            'merkle_root': self.merkle_root,
            'nonce': self.nonce
        }
        block_json = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_json.encode()).hexdigest()
    
    def mine_block(self, difficulty: int = 4) -> None:
        """Mine block with proof-of-work (for added security)."""
        target = "0" * difficulty
        self.merkle_root = self.compute_merkle_root()
        
        while not self.block_hash.startswith(target):
            self.nonce += 1
            self.block_hash = self.compute_block_hash()
        
        logger.info(f" Block {self.block_id} mined with nonce {self.nonce}")

class BlockchainAuditLog:
    """
    Blockchain-like audit log for immutable training data traceability.
    
    Creates tamper-evident chain of custody for all training data influences.
    """
    
    def __init__(self, log_dir: Union[str, Path], difficulty: int = 4):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.difficulty = difficulty
        
        # Initialize blockchain
        self.blocks: List[AuditBlock] = []
        self.current_entries: List[ImmutableLogEntry] = []
        self.max_entries_per_block = 100
        
        # Cryptographic keys for signing
        self.private_key = RSA.generate(2048)
        self.public_key = self.private_key.publickey()
        
        # Load existing blockchain or create genesis block
        self._load_or_create_genesis()
        
        logger.info(f" BlockchainAuditLog initialized with {len(self.blocks)} blocks")
    
    def _load_or_create_genesis(self):
        """Load existing blockchain or create genesis block."""
        blockchain_file = self.log_dir / "blockchain.json"
        
        if blockchain_file.exists():
            self._load_blockchain()
        else:
            self._create_genesis_block()
    
    def _create_genesis_block(self):
        """Create the first block in the blockchain."""
        genesis_entry = ImmutableLogEntry(
            entry_id="genesis",
            timestamp=time.time(),
            entry_type="genesis",
            dataset_id="system",
            affected_parameters=[],
            parameter_deltas={},
            gradient_norms={},
            data_provenance=DataProvenanceHash(
                dataset_id="system",
                data_hash="genesis",
                timestamp=time.time(),
                size_bytes=0
            ),
            metadata={"description": "Genesis block for CapibaraGPT v3 training audit"}
        )
        
        genesis_block = AuditBlock(
            block_id="genesis",
            previous_block_hash="0" * 64,
            timestamp=time.time(),
            entries=[genesis_entry],
            merkle_root=""
        )
        
        genesis_block.mine_block(self.difficulty)
        self.blocks.append(genesis_block)
        self._save_blockchain()
        
        logger.info(" Genesis block created and mined")
    
    def add_training_event(
        self,
        dataset_id: str,
        affected_parameters: List[str],
        parameter_deltas: Dict[str, float],
        gradient_norms: Dict[str, float],
        data_hash: str,
        data_size: int,
        source_url: Optional[str] = None,
        preprocessing_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a training event to the audit log."""
        
        # Create data provenance hash
        data_provenance = DataProvenanceHash(
            dataset_id=dataset_id,
            data_hash=data_hash,
            timestamp=time.time(),
            size_bytes=data_size,
            source_url=source_url,
            preprocessing_hash=preprocessing_hash
        )
        
        # Create immutable log entry
        entry = ImmutableLogEntry(
            entry_id=f"{dataset_id}_{int(time.time() * 1000000)}",
            timestamp=time.time(),
            entry_type="training_step",
            dataset_id=dataset_id,
            affected_parameters=affected_parameters,
            parameter_deltas=parameter_deltas,
            gradient_norms=gradient_norms,
            data_provenance=data_provenance,
            metadata=metadata or {}
        )
        
        self.current_entries.append(entry)
        
        # Create new block if we have enough entries
        if len(self.current_entries) >= self.max_entries_per_block:
            self._create_new_block()
        
        logger.debug(f" Training event added for dataset {dataset_id}")
        return entry.entry_id
    
    def _create_new_block(self):
        """Create and mine a new block with current entries."""
        if not self.current_entries:
            return
        
        previous_hash = self.blocks[-1].block_hash if self.blocks else "0" * 64
        
        new_block = AuditBlock(
            block_id=f"block_{len(self.blocks)}",
            previous_block_hash=previous_hash,
            timestamp=time.time(),
            entries=self.current_entries.copy(),
            merkle_root=""
        )
        
        new_block.mine_block(self.difficulty)
        self.blocks.append(new_block)
        self.current_entries.clear()
        
        # Save updated blockchain
        self._save_blockchain()
        
        logger.info(f"️ New block created: {new_block.block_id} with {len(new_block.entries)} entries")
    
    def verify_blockchain_integrity(self) -> Tuple[bool, List[str]]:
        """Verify the integrity of the entire blockchain."""
        issues = []
        
        if not self.blocks:
            return True, []
        
        # Verify genesis block
        if self.blocks[0].previous_block_hash != "0" * 64:
            issues.append("Genesis block has invalid previous hash")
        
        # Verify chain of blocks
        for i in range(1, len(self.blocks)):
            current_block = self.blocks[i]
            previous_block = self.blocks[i-1]
            
            # Verify previous hash link
            if current_block.previous_block_hash != previous_block.block_hash:
                issues.append(f"Block {i} has invalid previous hash link")
            
            # Verify block hash
            expected_hash = current_block.compute_block_hash()
            if current_block.block_hash != expected_hash:
                issues.append(f"Block {i} has invalid block hash")
            
            # Verify Merkle root
            expected_merkle = current_block.compute_merkle_root()
            if current_block.merkle_root != expected_merkle:
                issues.append(f"Block {i} has invalid Merkle root")
        
        is_valid = len(issues) == 0
        if is_valid:
            logger.info(" Blockchain integrity verified - no issues found")
        else:
            logger.error(f" Blockchain integrity issues: {issues}")
        
        return is_valid, issues
    
    def get_dataset_history(self, dataset_id: str) -> List[ImmutableLogEntry]:
        """Get all audit entries for a specific dataset."""
        entries = []
        
        for block in self.blocks:
            for entry in block.entries:
                if entry.dataset_id == dataset_id:
                    entries.append(entry)
        
        # Include current uncommitted entries
        for entry in self.current_entries:
            if entry.dataset_id == dataset_id:
                entries.append(entry)
        
        return sorted(entries, key=lambda x: x.timestamp)
    
    def get_parameter_lineage(self, parameter_name: str) -> List[ImmutableLogEntry]:
        """Get all audit entries that affected a specific parameter."""
        entries = []
        
        for block in self.blocks:
            for entry in block.entries:
                if parameter_name in entry.affected_parameters:
                    entries.append(entry)
        
        # Include current uncommitted entries
        for entry in self.current_entries:
            if parameter_name in entry.affected_parameters:
                entries.append(entry)
        
        return sorted(entries, key=lambda x: x.timestamp)
    
    def export_audit_report(self, output_path: Union[str, Path]) -> Dict[str, Any]:
        """Export comprehensive audit report."""
        # Force creation of final block
        if self.current_entries:
            self._create_new_block()
        
        # Verify integrity
        is_valid, issues = self.verify_blockchain_integrity()
        
        report = {
            "audit_metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_blocks": len(self.blocks),
                "total_entries": sum(len(block.entries) for block in self.blocks),
                "blockchain_valid": is_valid,
                "integrity_issues": issues
            },
            "dataset_summary": {},
            "parameter_summary": {},
            "chronological_events": []
        }
        
        # Analyze all entries
        all_entries = []
        for block in self.blocks:
            all_entries.extend(block.entries)
        
        # Dataset summary
        datasets = set()
        for entry in all_entries:
            datasets.add(entry.dataset_id)
            if entry.dataset_id not in report["dataset_summary"]:
                report["dataset_summary"][entry.dataset_id] = {
                    "total_events": 0,
                    "parameters_affected": set(),
                    "first_seen": entry.timestamp,
                    "last_seen": entry.timestamp
                }
            
            summary = report["dataset_summary"][entry.dataset_id]
            summary["total_events"] += 1
            summary["parameters_affected"].update(entry.affected_parameters)
            summary["first_seen"] = min(summary["first_seen"], entry.timestamp)
            summary["last_seen"] = max(summary["last_seen"], entry.timestamp)
        
        # Convert sets to lists for JSON serialization
        for dataset_id in report["dataset_summary"]:
            report["dataset_summary"][dataset_id]["parameters_affected"] = \
                list(report["dataset_summary"][dataset_id]["parameters_affected"])
        
        # Parameter summary
        parameters = set()
        for entry in all_entries:
            parameters.update(entry.affected_parameters)
        
        for param in parameters:
            param_entries = [e for e in all_entries if param in e.affected_parameters]
            datasets_affecting = set(e.dataset_id for e in param_entries)
            
            report["parameter_summary"][param] = {
                "total_updates": len(param_entries),
                "datasets_affecting": list(datasets_affecting),
                "first_update": min(e.timestamp for e in param_entries),
                "last_update": max(e.timestamp for e in param_entries)
            }
        
        # Chronological events (last 1000 for performance)
        recent_entries = sorted(all_entries, key=lambda x: x.timestamp)[-1000:]
        for entry in recent_entries:
            report["chronological_events"].append({
                "timestamp": entry.timestamp,
                "dataset_id": entry.dataset_id,
                "entry_type": entry.entry_type,
                "parameters_affected": len(entry.affected_parameters),
                "entry_hash": entry.compute_entry_hash()
            })
        
        # Save report
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f" Audit report exported to {output_path}")
        return report
    
    def _save_blockchain(self):
        """Save blockchain to disk."""
        blockchain_data = {
            "blocks": [
                {
                    "block_id": block.block_id,
                    "previous_block_hash": block.previous_block_hash,
                    "timestamp": block.timestamp,
                    "merkle_root": block.merkle_root,
                    "block_hash": block.block_hash,
                    "nonce": block.nonce,
                    "entries": [
                        {
                            "entry_id": entry.entry_id,
                            "timestamp": entry.timestamp,
                            "entry_type": entry.entry_type,
                            "dataset_id": entry.dataset_id,
                            "affected_parameters": entry.affected_parameters,
                            "parameter_deltas": entry.parameter_deltas,
                            "gradient_norms": entry.gradient_norms,
                            "data_provenance": {
                                "dataset_id": entry.data_provenance.dataset_id,
                                "data_hash": entry.data_provenance.data_hash,
                                "timestamp": entry.data_provenance.timestamp,
                                "size_bytes": entry.data_provenance.size_bytes,
                                "source_url": entry.data_provenance.source_url,
                                "preprocessing_hash": entry.data_provenance.preprocessing_hash
                            },
                            "metadata": entry.metadata
                        }
                        for entry in block.entries
                    ]
                }
                for block in self.blocks
            ]
        }
        
        blockchain_file = self.log_dir / "blockchain.json"
        with open(blockchain_file, 'w') as f:
            json.dump(blockchain_data, f, indent=2)
    
    def _load_blockchain(self):
        """Load blockchain from disk."""
        blockchain_file = self.log_dir / "blockchain.json"
        
        with open(blockchain_file, 'r') as f:
            blockchain_data = json.load(f)
        
        self.blocks = []
        for block_data in blockchain_data["blocks"]:
            entries = []
            for entry_data in block_data["entries"]:
                provenance_data = entry_data["data_provenance"]
                data_provenance = DataProvenanceHash(
                    dataset_id=provenance_data["dataset_id"],
                    data_hash=provenance_data["data_hash"],
                    timestamp=provenance_data["timestamp"],
                    size_bytes=provenance_data["size_bytes"],
                    source_url=provenance_data.get("source_url"),
                    preprocessing_hash=provenance_data.get("preprocessing_hash")
                )
                
                entry = ImmutableLogEntry(
                    entry_id=entry_data["entry_id"],
                    timestamp=entry_data["timestamp"],
                    entry_type=entry_data["entry_type"],
                    dataset_id=entry_data["dataset_id"],
                    affected_parameters=entry_data["affected_parameters"],
                    parameter_deltas=entry_data["parameter_deltas"],
                    gradient_norms=entry_data["gradient_norms"],
                    data_provenance=data_provenance,
                    metadata=entry_data["metadata"]
                )
                entries.append(entry)
            
            block = AuditBlock(
                block_id=block_data["block_id"],
                previous_block_hash=block_data["previous_block_hash"],
                timestamp=block_data["timestamp"],
                entries=entries,
                merkle_root=block_data["merkle_root"],
                block_hash=block_data["block_hash"],
                nonce=block_data["nonce"]
            )
            self.blocks.append(block)
        
        logger.info(f" Loaded blockchain with {len(self.blocks)} blocks")

# Factory function
def create_blockchain_audit_log(log_dir: str = "training_audit_logs") -> BlockchainAuditLog:
    """Factory function to create blockchain audit log."""
    return BlockchainAuditLog(log_dir)
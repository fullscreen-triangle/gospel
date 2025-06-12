"""
Knowledge distiller for Gospel LLM - focused on real genomic sequence data.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from gospel.llm.model import GospelLLM


class SequenceDataDistiller:
    """
    Sequence data distiller for LLM fine-tuning using real genomic sequences.
    
    This class implements the process of:
    1. Taking real genomic sequences from experiments
    2. Using expert analysis to understand their function
    3. Recording the sequence, analysis method, and biological interpretation
    4. Using this as training data for knowledge distillation
    """

    def __init__(
        self,
        expert_llm: GospelLLM,
        sequence_data_dir: Optional[str] = None,
    ):
        """
        Initialize a sequence data distiller.

        Args:
            expert_llm: Expert LLM instance (already trained)
            sequence_data_dir: Directory containing experimental sequence data
        """
        self.expert_llm = expert_llm
        self.sequence_data_dir = sequence_data_dir
        self.distillation_data = []

    def load_experimental_sequences(self, data_dir: str) -> List[Dict]:
        """
        Load real genomic sequences from experimental data.

        Args:
            data_dir: Directory containing sequence files

        Returns:
            List of sequence data dictionaries
        """
        sequences = []
        
        # This would load actual experimental sequence data
        # For now, we'll use a placeholder structure
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith(('.fasta', '.fa', '.seq')):
                    filepath = os.path.join(data_dir, filename)
                    with open(filepath, 'r') as f:
                        content = f.read()
                        sequences.append({
                            'filename': filename,
                            'sequence': content,
                            'source': 'experimental',
                            'filepath': filepath
                        })
        
        return sequences

    def analyze_sequence(self, sequence_data: Dict) -> Dict:
        """
        Analyze a genomic sequence using the expert LLM.

        Args:
            sequence_data: Dictionary containing sequence information

        Returns:
            Dictionary containing the analysis and biological interpretation
        """
        sequence = sequence_data['sequence']
        
        # Create a prompt for sequence analysis
        prompt = f"""
        Analyze the following genomic sequence and provide:
        1. Sequence composition and characteristics
        2. Potential functional regions or motifs
        3. Likely biological function or significance
        4. Comparison to known genomic elements
        
        Sequence: {sequence[:500]}{'...' if len(sequence) > 500 else ''}
        """
        
        # Get analysis from expert LLM
        analysis = self.expert_llm.query(prompt)
        
        return {
            'sequence': sequence,
            'analysis': analysis,
            'source_file': sequence_data.get('filename', 'unknown'),
            'sequence_length': len(sequence),
            'gc_content': (sequence.count('G') + sequence.count('C')) / len(sequence) if sequence else 0
        }

    def distill_from_sequences(
        self,
        sequence_data_dir: str,
        output_dir: str = "distillation_data"
    ) -> List[Dict]:
        """
        Generate training data from real genomic sequences.

        Args:
            sequence_data_dir: Directory containing experimental sequences
            output_dir: Directory to save distillation data

        Returns:
            List of distillation data records
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load experimental sequences
        sequences = self.load_experimental_sequences(sequence_data_dir)
        
        if not sequences:
            print(f"No sequence data found in {sequence_data_dir}")
            return []
        
        for i, seq_data in enumerate(tqdm(sequences, desc="Distilling from sequences")):
            # Analyze sequence
            analysis = self.analyze_sequence(seq_data)
            
            # Record distillation data
            data = {
                "id": f"sequence_{i+1}",
                "sequence": analysis['sequence'],
                "analysis": analysis['analysis'],
                "source_file": analysis['source_file'],
                "sequence_length": analysis['sequence_length'],
                "gc_content": analysis['gc_content'],
                "prompt_template": f"Analyze the genomic sequence and explain its biological significance: {analysis['sequence'][:100]}...",
            }
            
            self.distillation_data.append(data)
            
            # Save incrementally
            with open(os.path.join(output_dir, f"sequence_analysis_{i+1}.json"), "w") as f:
                json.dump(data, f, indent=2)
        
        # Save all data
        with open(os.path.join(output_dir, "all_sequence_distillation.json"), "w") as f:
            json.dump(self.distillation_data, f, indent=2)
        
        return self.distillation_data

    def prepare_training_data(
        self,
        output_dir: str = "training_data",
        format: str = "jsonl"
    ) -> str:
        """
        Prepare training data in a format suitable for fine-tuning.

        Args:
            output_dir: Directory to save training data
            format: Format of training data (jsonl, txt, etc.)

        Returns:
            Path to the training data file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.distillation_data:
            raise ValueError("No distillation data available. Run distill_from_sequences first.")
        
        if format == "jsonl":
            # Prepare for instruction fine-tuning format
            training_data = []
            for item in self.distillation_data:
                training_example = {
                    "instruction": f"Analyze this genomic sequence and explain its biological significance:",
                    "input": item["sequence"][:500] + "..." if len(item["sequence"]) > 500 else item["sequence"],
                    "output": item["analysis"]
                }
                training_data.append(training_example)
            
            # Save as JSONL
            output_file = os.path.join(output_dir, "sequence_training_data.jsonl")
            with open(output_file, "w") as f:
                for example in training_data:
                    f.write(json.dumps(example) + "\n")
            
            return output_file
        
        else:
            raise ValueError(f"Unsupported format: {format}")

    def export_for_ollama(self, output_dir: str = "ollama_model") -> str:
        """
        Export distilled knowledge for Ollama model creation.

        Args:
            output_dir: Directory to save Ollama model files

        Returns:
            Path to the Ollama model directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare training data
        training_file = self.prepare_training_data(output_dir)
        
        # Create Ollama Modelfile
        modelfile_content = f"""
FROM llama3

# Set parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

# Set system message
SYSTEM You are a genomic analysis expert specializing in sequence interpretation and biological significance.

# Add training data
"""
        
        # Add training examples to the Modelfile
        for item in self.distillation_data[:10]:  # Limit to first 10 for demo
            sequence_preview = item["sequence"][:200] + "..." if len(item["sequence"]) > 200 else item["sequence"]
            modelfile_content += f'\nTEMPLATE "Sequence: {sequence_preview}\\nAnalysis: {item["analysis"][:200]}..."\n'
        
        modelfile_path = os.path.join(output_dir, "Modelfile")
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)
        
        return output_dir


# Legacy compatibility
KnowledgeDistiller = SequenceDataDistiller 
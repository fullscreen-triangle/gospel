"""
Model trainer for Gospel LLM with systems biology integration.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from gospel.llm.distiller import KnowledgeDistiller


class ModelTrainer:
    """
    Trainer for Ollama models with domain-specific knowledge and systems biology integration.
    """

    def __init__(
        self,
        base_model: str = "llama3",
        model_name: str = "gospel-genomics",
        output_dir: str = "trained_model",
    ):
        """
        Initialize a model trainer.

        Args:
            base_model: Base Ollama model name
            model_name: Name for the trained model
            output_dir: Directory to save training outputs
        """
        self.base_model = base_model
        self.model_name = model_name
        self.output_dir = output_dir
        self.training_data = []
        self.network_data = {
            "proteomics": [],
            "reactomes": [],
            "interactomes": [],
            "pathways": [],
        }
        os.makedirs(output_dir, exist_ok=True)

    def add_training_data(self, data_path: str) -> None:
        """
        Add training data from a file.

        Args:
            data_path: Path to training data file (JSONL format)
        """
        with open(data_path, "r") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    self.training_data.append(item)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line: {line}")

    def add_distillation_data(self, distiller: KnowledgeDistiller) -> None:
        """
        Add training data from a knowledge distiller.

        Args:
            distiller: Knowledge distiller instance with distillation data
        """
        # Export distillation data in Ollama format
        export_dir = os.path.join(self.output_dir, "export")
        distiller.export_for_ollama(export_dir)
        
        # Add the exported data
        self.add_training_data(os.path.join(export_dir, "training.jsonl"))

    def add_network_data(self, data_type: str, data_path: str) -> None:
        """
        Add systems biology network data.

        Args:
            data_type: Type of network data (proteomics, reactomes, interactomes, pathways)
            data_path: Path to network data file (JSON or JSONL format)
        """
        if data_type not in self.network_data:
            raise ValueError(f"Unsupported network data type: {data_type}")
            
        if data_path.endswith('.json'):
            with open(data_path, "r") as f:
                data = json.load(f)
                self.network_data[data_type].append(data)
        elif data_path.endswith('.jsonl'):
            with open(data_path, "r") as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        self.network_data[data_type].append(item)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse line: {line}")
        else:
            print(f"Warning: Unsupported file format for {data_path}")

    def prepare_ollama_modelfile(self) -> str:
        """
        Prepare an Ollama Modelfile.

        Returns:
            Path to the Modelfile
        """
        # Create training data file if it doesn't exist
        if self.training_data:
            training_data_path = os.path.join(self.output_dir, "training.jsonl")
            with open(training_data_path, "w") as f:
                for item in self.training_data:
                    f.write(json.dumps(item) + "\n")
        
        # Prepare network data
        network_data_path = None
        if any(self.network_data.values()):
            network_data_path = os.path.join(self.output_dir, "network_data.json")
            with open(network_data_path, "w") as f:
                json.dump(self.network_data, f, indent=2)
        
        # Create Modelfile
        modelfile_path = os.path.join(self.output_dir, "Modelfile")
        with open(modelfile_path, "w") as f:
            f.write(f"FROM {self.base_model}\n")
            f.write("PARAMETER temperature 0.7\n")
            f.write("PARAMETER top_p 0.9\n")
            f.write(f"SYSTEM You are a genomics expert specializing in sprint performance analysis with deep knowledge of systems biology. You provide detailed and accurate information about genes, genetic variants, and their relationships to sprint performance and athletic ability. You understand both individual genes and the complex networks they participate in, including proteomics, reactomes, interactomes, metabolic pathways, and gene regulatory networks. You can explain how genes interact within biological systems and the emergent properties that arise from these interactions. Your analysis integrates multi-omics data to provide a comprehensive understanding of biological phenomena.\n")
            
            # Add template for completions
            f.write(f"TEMPLATE \"{{.System}}\\n\\n{{.Prompt}}\"\n")
            
            # Add training data if available
            if self.training_data:
                f.write(f"TRAINING_DATA training.jsonl\n")
        
        return modelfile_path

    def train(self, epochs: int = 3, learning_rate: float = 2e-5) -> None:
        """
        Train the model using Ollama.

        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for training
        """
        # Prepare Modelfile
        modelfile_path = self.prepare_ollama_modelfile()
        
        # Check if Ollama is installed and running
        try:
            subprocess.run(["ollama", "list"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: Ollama is not installed or not running.")
            print("Please install Ollama and start the service.")
            return
        
        # Create the model
        print(f"Creating Ollama model: {self.model_name}")
        create_cmd = ["ollama", "create", self.model_name, "-f", modelfile_path]
        
        try:
            result = subprocess.run(create_cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error creating model: {e}")
            print(e.stderr)
            return
        
        print(f"Model {self.model_name} created successfully")
        
        # Save model information
        model_info = {
            "name": self.model_name,
            "base_model": self.base_model,
            "training_examples": len(self.training_data),
            "network_data": {k: len(v) for k, v in self.network_data.items()},
            "modelfile_path": modelfile_path,
        }
        
        with open(os.path.join(self.output_dir, "model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model information saved to {os.path.join(self.output_dir, 'model_info.json')}")

    def export_model(self, export_path: Optional[str] = None) -> str:
        """
        Export the trained model.

        Args:
            export_path: Path to export the model (optional)

        Returns:
            Path to the exported model
        """
        if export_path is None:
            export_path = os.path.join(self.output_dir, f"{self.model_name}.tar")
        
        # Export the model using Ollama
        export_cmd = ["ollama", "export", self.model_name, export_path]
        
        try:
            result = subprocess.run(export_cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error exporting model: {e}")
            print(e.stderr)
            return ""
        
        print(f"Model exported to {export_path}")
        return export_path

    @staticmethod
    def import_model(model_path: str, model_name: Optional[str] = None) -> str:
        """
        Import a model from a file.

        Args:
            model_path: Path to the model file
            model_name: Name to give the imported model (optional)

        Returns:
            Name of the imported model
        """
        import_cmd = ["ollama", "import"]
        
        if model_name:
            import_cmd.extend([model_name, model_path])
        else:
            import_cmd.append(model_path)
        
        try:
            result = subprocess.run(import_cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            
            # Extract model name from output if not provided
            if not model_name:
                output = result.stdout.strip()
                # Try to extract model name from output
                if "imported" in output:
                    parts = output.split()
                    for i, part in enumerate(parts):
                        if part == "imported" and i > 0:
                            model_name = parts[i-1]
                            break
        except subprocess.CalledProcessError as e:
            print(f"Error importing model: {e}")
            print(e.stderr)
            return ""
        
        return model_name or "" 
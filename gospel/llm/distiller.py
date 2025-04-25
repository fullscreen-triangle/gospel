"""
Knowledge distiller for Gospel LLM.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from gospel.knowledge_base import KnowledgeBase
from gospel.llm.model import GospelLLM


class KnowledgeDistiller:
    """
    Knowledge distiller for LLM fine-tuning.
    
    This class implements the process of:
    1. Taking a problem or query
    2. Using a "solver" (expert LLM) to solve it
    3. Recording the prompt, solution method, and response
    4. Using this as training data for knowledge distillation
    """

    def __init__(
        self,
        expert_llm: GospelLLM,
        kb: Optional[KnowledgeBase] = None,
    ):
        """
        Initialize a knowledge distiller.

        Args:
            expert_llm: Expert LLM instance (already trained)
            kb: Knowledge base instance
        """
        self.expert_llm = expert_llm
        self.kb = kb or expert_llm.kb
        self.distillation_data = []

    def generate_problem(self, gene_name: Optional[str] = None) -> str:
        """
        Generate a problem related to a gene or genomics in general.

        Args:
            gene_name: Specific gene name (optional)

        Returns:
            Generated problem statement
        """
        if not self.kb:
            raise ValueError("Knowledge base is required for problem generation")
        
        if gene_name:
            gene_info = self.kb.get_gene_info(gene_name)
            prompt = f"""
            Generate a challenging but solvable genomics problem related to the gene {gene_name}.
            Use the following information about the gene:
            
            {gene_info}
            
            The problem should require understanding of this gene's function, variants, 
            and its role in systems biology networks, especially as related to sprint performance.
            """
        else:
            # Get a random set of genes from the knowledge base
            genes = list(self.kb.get_all_genes())[:5]
            gene_infos = [f"{gene}: {self.kb.get_gene_info(gene)[:100]}..." for gene in genes]
            genes_text = "\n".join(gene_infos)
            
            prompt = f"""
            Generate a challenging but solvable genomics problem related to sprint performance.
            Consider using information about these genes:
            
            {genes_text}
            
            The problem should require understanding of gene functions, variants,
            and systems biology networks, especially as related to sprint performance.
            """
        
        # Generate problem using the expert LLM
        problem = self.expert_llm.query(prompt)
        
        # Clean up and format the problem
        lines = problem.strip().split("\n")
        for i, line in enumerate(lines):
            if line.strip().lower().startswith(("problem:", "question:")):
                problem = line.split(":", 1)[1].strip()
                # Include any following lines that are part of the problem
                for j in range(i+1, len(lines)):
                    if lines[j].strip() and not lines[j].strip().lower().startswith(("hint:", "answer:", "solution:")):
                        problem += " " + lines[j].strip()
                    else:
                        break
                break
        
        return problem.strip()

    def solve_problem(self, problem: str) -> Dict:
        """
        Solve a problem using the expert LLM.

        Args:
            problem: Problem statement

        Returns:
            Dictionary containing the solution, method, and supporting information
        """
        return self.expert_llm.solve_genomic_problem(problem)

    def distill_knowledge(
        self,
        num_problems: int = 50,
        genes: Optional[List[str]] = None,
        output_dir: str = "distillation_data"
    ) -> List[Dict]:
        """
        Generate problems, solve them, and record the distillation data.

        Args:
            num_problems: Number of problems to generate
            genes: List of genes to focus on (optional)
            output_dir: Directory to save distillation data

        Returns:
            List of distillation data records
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if not genes and self.kb:
            all_genes = list(self.kb.get_all_genes())
            # Use a subset if there are too many
            genes = all_genes[:min(len(all_genes), num_problems * 2)]
        
        for i in tqdm(range(num_problems), desc="Distilling knowledge"):
            # Select a gene if provided, otherwise generate generic problem
            gene = None
            if genes:
                gene = genes[i % len(genes)]
            
            # Generate problem
            problem = self.generate_problem(gene)
            
            # Solve problem
            solution = self.solve_problem(problem)
            
            # Record distillation data
            data = {
                "id": f"problem_{i+1}",
                "problem": problem,
                "solution": solution["solution"],
                "method": solution["method"],
                "genes": solution["genes"],
                "prompt_template": f"Solve the following genomics problem: {problem}",
                "networks": solution.get("networks", {}),
            }
            
            self.distillation_data.append(data)
            
            # Save incrementally
            with open(os.path.join(output_dir, f"distillation_data_{i+1}.json"), "w") as f:
                json.dump(data, f, indent=2)
        
        # Save all data
        with open(os.path.join(output_dir, "all_distillation_data.json"), "w") as f:
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
            raise ValueError("No distillation data available. Run distill_knowledge first.")
        
        if format == "jsonl":
            # Prepare for instruction fine-tuning format
            training_data = []
            for item in self.distillation_data:
                training_example = {
                    "instruction": item["prompt_template"],
                    "input": "",  # Empty input as the instruction contains the problem
                    "output": item["solution"]
                }
                training_data.append(training_example)
            
            output_path = os.path.join(output_dir, "training_data.jsonl")
            with open(output_path, "w") as f:
                for item in training_data:
                    f.write(json.dumps(item) + "\n")
        
        elif format == "txt":
            # Simple text format with separators
            output_path = os.path.join(output_dir, "training_data.txt")
            with open(output_path, "w") as f:
                for item in self.distillation_data:
                    f.write(f"### INSTRUCTION:\n{item['prompt_template']}\n\n")
                    f.write(f"### RESPONSE:\n{item['solution']}\n\n")
                    f.write("### END\n\n")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return output_path

    def export_for_ollama(self, output_dir: str = "ollama_model") -> str:
        """
        Export distillation data in a format suitable for Ollama fine-tuning.

        Args:
            output_dir: Directory to save Ollama fine-tuning data

        Returns:
            Path to the Ollama model definition file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.distillation_data:
            raise ValueError("No distillation data available. Run distill_knowledge first.")
        
        # Prepare training data in Ollama format
        training_data_path = os.path.join(output_dir, "training.jsonl")
        with open(training_data_path, "w") as f:
            for item in self.distillation_data:
                training_example = {
                    "prompt": item["prompt_template"],
                    "response": item["solution"]
                }
                f.write(json.dumps(training_example) + "\n")
        
        # Create Modelfile
        modelfile_path = os.path.join(output_dir, "Modelfile")
        with open(modelfile_path, "w") as f:
            f.write(f"FROM {self.expert_llm.base_model}\n")
            f.write("PARAMETER temperature 0.7\n")
            f.write("PARAMETER top_p 0.9\n")
            f.write(f"SYSTEM You are a genomics expert specializing in sprint performance analysis.\n")
            f.write(f"TEMPLATE \"{{.System}}\\n\\n{{.Prompt}}\"\n")
            f.write(f"TRAINING_DATA training.jsonl\n")
        
        return modelfile_path 
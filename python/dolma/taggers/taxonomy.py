"""

Taxonomy Classifier.

This tagger uses EssentialAI/EAI-Distill-0.5b to classify content based on taxonomy.
It samples chunks from long documents and provides a classification result.

"""

import random
from typing import Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..core.data_types import DocResult, Document, Span, TextSlice
from ..core.registry import TaggerRegistry
from ..core.taggers import BaseTagger


@TaggerRegistry.add("taxonomy")
class TaxonomyClassifier(BaseTagger):
    def __init__(self):
        super().__init__()
        self.model_name = "EssentialAI/EAI-Distill-0.5b"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.max_char_per_doc = 30000
    
    @property
    def defaults(self) -> List[str]:
        return ["taxonomy_classification"]
    
    def chunk_text(self, text):
        """Sample chunks from long documents to stay within token limits."""
        if len(text) <= self.max_char_per_doc:
            return text
            
        chunk_size = self.max_char_per_doc // 3
        start = text[:chunk_size]
        
        middle_start = chunk_size 
        middle_end = len(text) - chunk_size 
        
        mid_point = random.randint(middle_start + chunk_size//2, middle_end - chunk_size//2)
        
        middle = text[mid_point - chunk_size//2:mid_point + chunk_size//2]
        end = text[-chunk_size:]
        return f"[beginning]\n{start}\n[middle]\n{middle}\n[end]\n{end}"
    
    def classify_document(self, text: str) -> str:
        """Classify the document using the EAI-Distill model."""
        chunked_text = self.chunk_text(text)
        
        messages = [
            {"role": "system", "content": "taxonomy"},
            {"role": "user", "content": chunked_text},
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    @property
    def defaults(self) -> List[str]:
        return [
            "fdc_primary", "fdc_secondary", 
            "fdc_primary_value", "fdc_secondary_value",
            "bloom_cognitive_primary", "bloom_cognitive_secondary",
            "bloom_cognitive_primary_value", "bloom_cognitive_secondary_value",
            "bloom_knowledge_primary", "bloom_knowledge_secondary",
            "bloom_knowledge_primary_value", "bloom_knowledge_secondary_value",
            "doc_type_primary", "doc_type_secondary",
            "doc_type_primary_value", "doc_type_secondary_value",
            "extraction_artifacts_primary", "extraction_artifacts_secondary",
            "extraction_artifacts_primary_value", "extraction_artifacts_secondary_value",
            "missing_content_primary", "missing_content_secondary",
            "missing_content_primary_value", "missing_content_secondary_value",
            "doc_type_v2_primary", "doc_type_v2_secondary",
            "doc_type_v2_primary_value", "doc_type_v2_secondary_value",
            "reasoning_depth_primary", "reasoning_depth_secondary",
            "reasoning_depth_primary_value", "reasoning_depth_secondary_value",
            "technical_correctness_primary", "technical_correctness_secondary",
            "technical_correctness_primary_value", "technical_correctness_secondary_value",
            "education_level_primary", "education_level_secondary",
            "education_level_primary_value", "education_level_secondary_value"
        ]
    
    def parse_classification(self, text: str) -> Dict[str, Tuple[str, float, str, float]]:
        """Parse the classification output from the model."""
        # Extract the actual model output (remove the input prompt)
        lines = text.split('\n')
        
        # Define the expected taxonomy labels
        labels = [
            "FDC (Dewey)", "Bloom cognitive", "Bloom knowledge", "Doc type",
            "Extraction artifacts", "Missing content", "Doc type v2",
            "Reasoning depth", "Technical correctness", "Education level"
        ]
        
        # Map the labels to attribute names
        attribute_mapping = {
            "FDC (Dewey)": "fdc",
            "Bloom cognitive": "bloom_cognitive", 
            "Bloom knowledge": "bloom_knowledge",
            "Doc type": "doc_type",
            "Extraction artifacts": "extraction_artifacts",
            "Missing content": "missing_content",
            "Doc type v2": "doc_type_v2",
            "Reasoning depth": "reasoning_depth",
            "Technical correctness": "technical_correctness",
            "Education level": "education_level"
        }
        
        # Find where the response starts (after the input)
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip() == "ASSISTANT:" or line.strip() == "<answer>":
                start_idx = i + 1
                break
        
        # Extract classification lines
        classification_lines = lines[start_idx:start_idx+len(labels)]
        
        # Parse classifications
        classifications = {}
        for i, label in enumerate(labels):
            if i < len(classification_lines):
                values = classification_lines[i].strip().split(',')
                primary = values[0].strip() if values and len(values) > 0 else "0"
                secondary = values[1].strip() if values and len(values) > 1 else "0"
                
                # Try to convert primary to float for classification score
                try:
                    primary_value = float(primary)
                except (ValueError, TypeError):
                    primary_value = 0.0
                
                # The confidence is either 1.0 or based on model output quality
                primary_score = 1.0
                
                # Try to convert secondary to float for classification score
                try:
                    secondary_value = float(secondary)
                except (ValueError, TypeError):
                    secondary_value = 0.0
                
                # The confidence is either 1.0 or based on model output quality
                secondary_score = 1.0
                
                attr_name = attribute_mapping[label]
                classifications[attr_name] = (primary, primary_score, secondary, secondary_score, primary_value, secondary_value)
            else:
                attr_name = attribute_mapping[label]
                classifications[attr_name] = ("0", 0.0, "0", 0.0, 0.0, 0.0)
        
        return classifications
    
    def predict(self, doc: Document) -> DocResult:
        """Predict taxonomy classifications for a document."""
        classification_text = self.classify_document(doc.text)
        classifications = self.parse_classification(classification_text)
        
        spans = []
        doc_length = len(doc.text)
        
        # Create spans for each classification
        for attr_name, (primary, primary_score, secondary, secondary_score, primary_value, secondary_value) in classifications.items():
            # Add primary span
            spans.append(
                Span(
                    start=0,
                    end=doc_length,
                    type=f"{attr_name}_primary",
                    score=primary_score
                )
            )
            
            # Store the actual classification value as a score in a value span
            value_span_type = f"{attr_name}_primary_value"
            spans.append(
                Span(
                    start=0,
                    end=doc_length,
                    type=value_span_type,
                    score=primary_value  # Store the numeric value directly as score
                )
            )
            
            # Add secondary span
            spans.append(
                Span(
                    start=0,
                    end=doc_length,
                    type=f"{attr_name}_secondary",
                    score=secondary_score
                )
            )
            
            # Store the secondary classification value as a score in a value span
            value_span_type = f"{attr_name}_secondary_value"
            spans.append(
                Span(
                    start=0,
                    end=doc_length,
                    type=value_span_type,
                    score=secondary_value  # Store the numeric value directly as score
                )
            )
        
        return DocResult(doc=doc, spans=spans)
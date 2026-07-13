"""
================================================================================
probe/hidden_state_extractor.py
================================================================================

Extracts hidden states, attention weights, token/positional embeddings, 
and output distributions from Anchor and Student models.
"""

import torch
import logging
from typing import Dict, Any, List, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger("probe.hidden_state_extractor")

class HiddenStateExtractor:
    """
    Orchestrates the forward passes to extract representation tensors 
    from the models in a memory-safe, CPU-accumulated manner.
    """
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def extract_features(self, prompts: List[str], batch_size: int = 8) -> Dict[str, Any]:
        """
        Runs the model over prompts in batches and extracts all target representations.
        Tensors are moved to CPU immediately to avoid GPU memory growth.
        
        Returns:
            Dict containing:
                - 'token_embeddings': List[torch.Tensor]
                - 'position_embeddings': List[torch.Tensor]
                - 'hidden_states': List[Tuple[torch.Tensor, ...]] (one tuple per prompt)
                - 'attention_weights': List[Tuple[torch.Tensor, ...]] (one tuple per prompt)
                - 'logits': List[torch.Tensor]
                - 'predicted_tokens': List[torch.Tensor]
        """
        self.model.eval()
        
        extracted = {
            "token_embeddings": [],
            "position_embeddings": [],
            "hidden_states": [],       # Outer list: batch/prompt, Inner: tuple of layer outputs
            "attention_weights": [],   # Outer list: batch/prompt, Inner: tuple of layer attention weights
            "logits": [],
            "predicted_tokens": []
        }
        
        logger.info("Starting representation extraction for %d prompts (batch_size=%d)...", len(prompts), batch_size)
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Tokenize batch. Pad to longest sequence in this batch
            inputs = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            batch_len = input_ids.size(0)
            seq_len = input_ids.size(1)
            
            # Calculate position IDs
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
            position_ids = position_ids.expand(batch_len, -1)
            
            # Explicitly force configuration settings in case library version ignores kwargs
            self.model.config.output_attentions = True
            self.model.config.output_hidden_states = True

            with torch.no_grad():
                # Extract embeddings directly using standard GPT2 structure
                # transformer.wte: word token embeddings
                # transformer.wpe: word position embeddings
                wte = self.model.transformer.wte(input_ids).cpu()
                wpe = self.model.transformer.wpe(position_ids).cpu()
                
                # Run complete forward pass with outputs enabled
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict=True
                )
                
                # Debug prints as requested
                print(f"DEBUG: len(outputs.attentions) = {len(outputs.attentions) if outputs.attentions is not None else 'None'}")
                print(f"DEBUG: type(outputs.attentions) = {type(outputs.attentions)}")
                
                logits = outputs.logits.cpu()
                predicted_tokens = torch.argmax(outputs.logits, dim=-1).cpu()
                
                # outputs.hidden_states is a tuple of length 7: embedding + 6 transformer blocks
                # Move to cpu
                batch_hidden_states = [hs.cpu() for hs in outputs.hidden_states]
                
                # outputs.attentions is a tuple of length 6: attention weights for each block
                # Each is [batch, num_heads, seq_len, seq_len]
                batch_attentions = [att.cpu() for att in outputs.attentions]
                
            # Disassemble batch to store individually per prompt
            for b in range(batch_len):
                # Retrieve actual non-padded length
                actual_len = attention_mask[b].sum().item()
                
                extracted["token_embeddings"].append(wte[b, :actual_len])
                extracted["position_embeddings"].append(wpe[b, :actual_len])
                extracted["logits"].append(logits[b, :actual_len])
                extracted["predicted_tokens"].append(predicted_tokens[b, :actual_len])
                
                # Slice hidden states and attentions to ignore pad tokens
                prompt_hs = tuple(hs[b, :actual_len] for hs in batch_hidden_states)
                extracted["hidden_states"].append(prompt_hs)
                
                # Attention maps are 4D: [batch, heads, seq, seq]
                prompt_att = tuple(att[b, :, :actual_len, :actual_len] for att in batch_attentions)
                extracted["attention_weights"].append(prompt_att)
                
        logger.info("Successfully completed representation extraction.")
        return extracted

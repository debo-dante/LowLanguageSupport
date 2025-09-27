"""
Indian Language Model

Transformer-based language model optimized for Indian languages,
with support for multilingual learning and low-resource scenarios.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    BertModel, BertConfig, BertTokenizer,
    RobertaModel, RobertaConfig, RobertaTokenizer,
    GPT2Model, GPT2Config, GPT2Tokenizer,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Tuple, Dict, List, Union
import numpy as np
import logging
from pathlib import Path
import json


class IndianLanguageModel(nn.Module):
    """
    Transformer-based language model optimized for Indian languages.
    
    Supports various architectures (BERT, RoBERTa, GPT-2) and can be
    initialized from pretrained models or trained from scratch.
    """
    
    def __init__(self,
                 language: str,
                 model_type: str = 'bert',
                 vocab_size: int = 50000,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 max_position_embeddings: int = 512,
                 dropout_prob: float = 0.1,
                 pretrained_model_name: Optional[str] = None,
                 custom_config: Optional[Dict] = None):
        """
        Initialize Indian Language Model.
        
        Args:
            language: Target language code
            model_type: Type of model ('bert', 'roberta', 'gpt2', 'custom')
            vocab_size: Vocabulary size
            hidden_size: Hidden dimension size
            num_hidden_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            intermediate_size: Feed-forward network intermediate size
            max_position_embeddings: Maximum sequence length
            dropout_prob: Dropout probability
            pretrained_model_name: Name of pretrained model to load
            custom_config: Custom configuration dictionary
        """
        super().__init__()
        
        # Setup logging first
        self.logger = logging.getLogger(__name__)
        
        self.language = language
        self.model_type = model_type
        self.config = self._create_config(
            model_type, vocab_size, hidden_size, num_hidden_layers,
            num_attention_heads, intermediate_size, max_position_embeddings,
            dropout_prob, custom_config
        )
        
        # Initialize tokenizer and model
        if pretrained_model_name:
            self._load_pretrained(pretrained_model_name)
        else:
            self._initialize_from_scratch()
        
        # Language-specific adaptations
        self._setup_language_adaptations()
    
    def _create_config(self, model_type: str, vocab_size: int, hidden_size: int,
                      num_hidden_layers: int, num_attention_heads: int,
                      intermediate_size: int, max_position_embeddings: int,
                      dropout_prob: float, custom_config: Optional[Dict]) -> AutoConfig:
        """Create model configuration."""
        
        base_config = {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_hidden_layers': num_hidden_layers,
            'num_attention_heads': num_attention_heads,
            'intermediate_size': intermediate_size,
            'max_position_embeddings': max_position_embeddings,
            'hidden_dropout_prob': dropout_prob,
            'attention_probs_dropout_prob': dropout_prob,
            'type_vocab_size': 2,
            'initializer_range': 0.02,
            'layer_norm_eps': 1e-12,
            'pad_token_id': 0,
        }
        
        if custom_config:
            base_config.update(custom_config)
        
        if model_type == 'bert':
            return BertConfig(**base_config)
        elif model_type == 'roberta':
            base_config.update({
                'bos_token_id': 0,
                'eos_token_id': 2,
                'sep_token_id': 2,
                'cls_token_id': 0,
                'mask_token_id': 4
            })
            return RobertaConfig(**base_config)
        elif model_type == 'gpt2':
            base_config.update({
                'n_positions': max_position_embeddings,
                'n_ctx': max_position_embeddings,
                'n_embd': hidden_size,
                'n_layer': num_hidden_layers,
                'n_head': num_attention_heads,
                'resid_pdrop': dropout_prob,
                'embd_pdrop': dropout_prob,
                'attn_pdrop': dropout_prob,
                'bos_token_id': 50256,
                'eos_token_id': 50256,
            })
            return GPT2Config(**base_config)
        else:
            # Use BERT config as default for custom models
            return BertConfig(**base_config)
    
    def _load_pretrained(self, model_name: str):
        """Load pretrained model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer = AutoModel.from_pretrained(model_name)
            self.config = self.transformer.config
            self.logger.info(f"Loaded pretrained model: {model_name}")
        except Exception as e:
            self.logger.warning(f"Failed to load pretrained model {model_name}: {e}")
            self.logger.info("Initializing from scratch instead")
            self._initialize_from_scratch()
    
    def _initialize_from_scratch(self):
        """Initialize model from scratch."""
        if self.model_type == 'bert':
            self.transformer = BertModel(self.config)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif self.model_type == 'roberta':
            self.transformer = RobertaModel(self.config)
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif self.model_type == 'gpt2':
            self.transformer = GPT2Model(self.config)
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            # Default to BERT-like architecture
            self.transformer = BertModel(self.config)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        self.logger.info(f"Initialized {self.model_type} model from scratch")
    
    def _setup_language_adaptations(self):
        """Setup language-specific adaptations."""
        # Add language-specific improvements
        
        # 1. Position embeddings for longer sequences common in Indian languages
        if hasattr(self.transformer, 'embeddings'):
            if hasattr(self.transformer.embeddings, 'position_embeddings'):
                pos_emb = self.transformer.embeddings.position_embeddings
                if pos_emb.num_embeddings < 1024:
                    # Extend position embeddings for longer sequences
                    self._extend_position_embeddings(1024)
        
        # 2. Add script-specific embedding layer for multilingual support
        self.script_embeddings = nn.Embedding(10, self.config.hidden_size)  # Support up to 10 scripts
        
        # 3. Language-specific normalization layer
        self.language_norm = nn.LayerNorm(self.config.hidden_size)
        
        # 4. Adapter layers for fine-tuning efficiency
        self.adapter_down = nn.Linear(self.config.hidden_size, self.config.hidden_size // 4)
        self.adapter_up = nn.Linear(self.config.hidden_size // 4, self.config.hidden_size)
        self.adapter_dropout = nn.Dropout(0.1)
    
    def _extend_position_embeddings(self, new_max_length: int):
        """Extend position embeddings to handle longer sequences."""
        old_pos_emb = self.transformer.embeddings.position_embeddings
        old_max_length = old_pos_emb.num_embeddings
        
        if new_max_length <= old_max_length:
            return
        
        # Create new position embeddings
        new_pos_emb = nn.Embedding(new_max_length, self.config.hidden_size)
        
        # Copy old weights
        with torch.no_grad():
            new_pos_emb.weight[:old_max_length] = old_pos_emb.weight
            # Initialize new positions by copying the last position
            for i in range(old_max_length, new_max_length):
                new_pos_emb.weight[i] = old_pos_emb.weight[-1]
        
        # Replace old embeddings
        self.transformer.embeddings.position_embeddings = new_pos_emb
        self.config.max_position_embeddings = new_max_length
        
        self.logger.info(f"Extended position embeddings to {new_max_length}")
    
    def get_script_id(self, language: str) -> int:
        """Get script ID for a language."""
        script_mapping = {
            'hi': 0,  # Devanagari
            'bn': 1,  # Bengali
            'ta': 2,  # Tamil
            'te': 3,  # Telugu
            'mr': 0,  # Marathi (Devanagari)
            'gu': 4,  # Gujarati
            'pa': 5,  # Punjabi (Gurmukhi)
            'or': 6,  # Odia
            'ml': 7,  # Malayalam
            'kn': 8,  # Kannada
            'as': 1,  # Assamese (Bengali script)
        }
        return script_mapping.get(language, 9)  # Unknown script
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                language_ids: Optional[torch.Tensor] = None,
                return_dict: bool = True) -> Union[Tuple, BaseModelOutput]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            position_ids: Position IDs
            inputs_embeds: Input embeddings (alternative to input_ids)
            language_ids: Language/script IDs for multilingual support
            return_dict: Whether to return a dictionary
            
        Returns:
            Model outputs
        """
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True
        )
        
        hidden_states = transformer_outputs.last_hidden_state
        
        # Add script-specific embeddings if language_ids provided
        if language_ids is not None:
            if language_ids.dim() == 1:
                # Single language ID per sequence
                script_embeds = self.script_embeddings(language_ids).unsqueeze(1)
                script_embeds = script_embeds.expand(-1, hidden_states.size(1), -1)
            else:
                # Token-level language IDs
                script_embeds = self.script_embeddings(language_ids)
            
            hidden_states = hidden_states + script_embeds
        
        # Apply language-specific normalization
        hidden_states = self.language_norm(hidden_states)
        
        # Apply adapter layers (for efficient fine-tuning)
        adapter_output = self.adapter_down(hidden_states)
        adapter_output = F.relu(adapter_output)
        adapter_output = self.adapter_dropout(adapter_output)
        adapter_output = self.adapter_up(adapter_output)
        
        # Residual connection
        hidden_states = hidden_states + adapter_output
        
        if return_dict:
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions
            )
        
        return (hidden_states,) + transformer_outputs[1:]
    
    def get_embeddings(self, text: Union[str, List[str]], 
                      language: Optional[str] = None) -> torch.Tensor:
        """
        Get embeddings for input text.
        
        Args:
            text: Input text(s)
            language: Language code for script-specific embeddings
            
        Returns:
            Text embeddings
        """
        # Tokenize input
        if isinstance(text, str):
            text = [text]
        
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Prepare language IDs if provided
        language_ids = None
        if language:
            script_id = self.get_script_id(language)
            language_ids = torch.full((len(text),), script_id, dtype=torch.long)
        
        # Forward pass
        with torch.no_grad():
            outputs = self(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                language_ids=language_ids
            )
        
        # Return mean pooled embeddings
        embeddings = outputs.last_hidden_state
        attention_mask = encoded['attention_mask'].unsqueeze(-1)
        
        # Mean pooling with attention mask
        embeddings = (embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        
        return embeddings
    
    def save_model(self, save_path: Union[str, Path]):
        """Save model and tokenizer."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save transformer model
        self.transformer.save_pretrained(save_path / "transformer")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path / "tokenizer")
        
        # Save custom components
        custom_state = {
            'script_embeddings': self.script_embeddings.state_dict(),
            'language_norm': self.language_norm.state_dict(),
            'adapter_down': self.adapter_down.state_dict(),
            'adapter_up': self.adapter_up.state_dict(),
            'language': self.language,
            'model_type': self.model_type
        }
        
        torch.save(custom_state, save_path / "custom_components.pt")
        
        # Save configuration
        config_dict = {
            'language': self.language,
            'model_type': self.model_type,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config)
        }
        
        with open(save_path / "model_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def load_model(cls, load_path: Union[str, Path]):
        """Load saved model."""
        load_path = Path(load_path)
        
        # Load configuration
        with open(load_path / "model_config.json", 'r') as f:
            config_dict = json.load(f)
        
        # Initialize model
        model = cls(
            language=config_dict['language'],
            model_type=config_dict['model_type']
        )
        
        # Load transformer and tokenizer
        model.transformer = AutoModel.from_pretrained(load_path / "transformer")
        model.tokenizer = AutoTokenizer.from_pretrained(load_path / "tokenizer")
        
        # Load custom components
        custom_state = torch.load(load_path / "custom_components.pt")
        model.script_embeddings.load_state_dict(custom_state['script_embeddings'])
        model.language_norm.load_state_dict(custom_state['language_norm'])
        model.adapter_down.load_state_dict(custom_state['adapter_down'])
        model.adapter_up.load_state_dict(custom_state['adapter_up'])
        
        return model
    
    def freeze_transformer(self):
        """Freeze transformer parameters for efficient fine-tuning."""
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        self.logger.info("Froze transformer parameters")
    
    def unfreeze_transformer(self):
        """Unfreeze transformer parameters."""
        for param in self.transformer.parameters():
            param.requires_grad = True
        
        self.logger.info("Unfroze transformer parameters")
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count breakdown."""
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        script_emb_params = sum(p.numel() for p in self.script_embeddings.parameters())
        adapter_params = (sum(p.numel() for p in self.adapter_down.parameters()) +
                         sum(p.numel() for p in self.adapter_up.parameters()))
        norm_params = sum(p.numel() for p in self.language_norm.parameters())
        
        total_params = transformer_params + script_emb_params + adapter_params + norm_params
        
        return {
            'total': total_params,
            'transformer': transformer_params,
            'script_embeddings': script_emb_params,
            'adapters': adapter_params,
            'normalization': norm_params
        }

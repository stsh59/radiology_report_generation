"""
Evaluation metrics for medical report generation.
"""
import numpy as np
import torch
from typing import List, Dict
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import logging
import nltk

logger = logging.getLogger(__name__)

# Download required NLTK data for METEOR
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK 'punkt'...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading NLTK 'wordnet'...")
    nltk.download('wordnet', quiet=True)


class MedicalReportMetrics:
    """
    Comprehensive metrics for medical report generation.
    """
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
    
    def compute_bleu(self, references: List[List[str]], hypotheses: List[str]) -> Dict[str, float]:
        """
        Compute BLEU scores.
        
        Args:
            references: List of reference texts (each text is tokenized into words)
            hypotheses: List of generated texts (each tokenized into words)
        
        Returns:
            Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
        """
        refs_tokenized = []
        hyps_tokenized = []
        
        for ref in references:
            refs_tokenized.append([ref[0].split()])
        
        for hyp in hypotheses:
            hyps_tokenized.append(hyp.split())
        
        bleu_scores = {}
        for n in range(1, 5):
            weights = tuple([1.0/n] * n + [0.0] * (4-n))
            bleu_scores[f'bleu-{n}'] = corpus_bleu(
                refs_tokenized,
                hyps_tokenized,
                weights=weights,
                smoothing_function=self.smoothing
            )
        
        return bleu_scores
    
    def compute_rouge(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            references: List of reference texts
            hypotheses: List of generated texts
        
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for ref, hyp in zip(references, hypotheses):
            scores = self.rouge_scorer.score(ref, hyp)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        return {
            'rouge-1': np.mean(rouge_scores['rouge1']),
            'rouge-2': np.mean(rouge_scores['rouge2']),
            'rouge-L': np.mean(rouge_scores['rougeL'])
        }
    
    def compute_meteor(self, references: List[str], hypotheses: List[str]) -> float:
        """
        Compute METEOR score using NLTK (no Java dependency).
        
        Args:
            references: List of reference texts
            hypotheses: List of generated texts
        
        Returns:
            METEOR score
        """
        try:
            from nltk.translate.meteor_score import meteor_score
            from nltk import word_tokenize
        except ImportError:
            logger.warning("NLTK meteor_score not available")
            return 0.0
        
        meteor_scores = []
        
        for ref, hyp in zip(references, hypotheses):
            try:
                ref_tokens = word_tokenize(ref.lower())
                hyp_tokens = word_tokenize(hyp.lower())
                score = meteor_score([ref_tokens], hyp_tokens)
                meteor_scores.append(score)
            except Exception as e:
                logger.warning(f"METEOR computation failed for a sample: {e}")
                meteor_scores.append(0.0)
        
        return np.mean(meteor_scores) if meteor_scores else 0.0
    
    def compute_bertscore(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """
        Compute BERTScore for semantic similarity.
        
        Args:
            references: List of reference texts
            hypotheses: List of generated texts
        
        Returns:
            Dictionary with BERTScore precision, recall, and F1
        """
        try:
            from bert_score import score
            P, R, F1 = score(hypotheses, references, lang="en", rescale_with_baseline=True, verbose=False)
            return {
                "bertscore_precision": P.mean().item(),
                "bertscore_recall": R.mean().item(),
                "bertscore_f1": F1.mean().item()
            }
        except Exception as e:
            logger.warning(f"Failed to compute BERTScore: {e}")
            return {
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0
            }
    
    def compute_chexbert_f1(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """
        Compute CheXbert F1 for clinical accuracy.
        
        Extracts 14 pathology labels from both reference and generated reports,
        then computes F1 scores to measure clinical correctness.
        
        Args:
            references: List of reference texts
            hypotheses: List of generated texts
        
        Returns:
            Dictionary with CheXbert accuracy, micro F1, macro F1
        """
        try:
            from eval.chexbert_labeler import CheXbertLabeler, compute_chexbert_metrics
            
            labeler = CheXbertLabeler(use_gpu=torch.cuda.is_available())
            
            ref_labels = labeler.extract_labels(references)
            gen_labels = labeler.extract_labels(hypotheses)
            
            metrics = compute_chexbert_metrics(ref_labels, gen_labels)
            
            # Return main metrics (exclude per-class for summary)
            return {
                "chexbert_accuracy": metrics["chexbert_accuracy"],
                "chexbert_f1_micro": metrics["chexbert_f1_micro"],
                "chexbert_f1_macro": metrics["chexbert_f1_macro"],
                "chexbert_precision": metrics["chexbert_precision_micro"],
                "chexbert_recall": metrics["chexbert_recall_micro"]
            }
        except Exception as e:
            logger.warning(f"Failed to compute CheXbert F1: {e}")
            return {
                "chexbert_accuracy": 0.0,
                "chexbert_f1_micro": 0.0,
                "chexbert_f1_macro": 0.0,
                "chexbert_precision": 0.0,
                "chexbert_recall": 0.0
            }
    
    def compute_all_metrics(
        self,
        references: List[str],
        hypotheses: List[str]
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            references: List of reference texts
            hypotheses: List of generated texts
        
        Returns:
            Dictionary with all metrics
        """
        all_metrics = {}
        
        refs_for_bleu = [[ref] for ref in references]
        bleu_scores = self.compute_bleu(refs_for_bleu, hypotheses)
        all_metrics.update(bleu_scores)
        
        rouge_scores = self.compute_rouge(references, hypotheses)
        all_metrics.update(rouge_scores)
        
        try:
            meteor_score = self.compute_meteor(references, hypotheses)
            all_metrics['meteor'] = meteor_score
        except Exception as e:
            logger.warning(f"Failed to compute METEOR score: {e}")
            all_metrics['meteor'] = 0.0
        
        # BERTScore (semantic similarity)
        bertscore_metrics = self.compute_bertscore(references, hypotheses)
        all_metrics.update(bertscore_metrics)
        
        # CheXbert F1 (clinical accuracy)
        chexbert_metrics = self.compute_chexbert_f1(references, hypotheses)
        all_metrics.update(chexbert_metrics)
        
        return all_metrics


def compute_cosine_similarity(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor
) -> float:
    """
    Compute average cosine similarity between two sets of embeddings.
    
    Args:
        embeddings1: First set of embeddings [N, D]
        embeddings2: Second set of embeddings [N, D]
    
    Returns:
        Average cosine similarity
    """
    embeddings1_norm = embeddings1 / embeddings1.norm(dim=-1, keepdim=True)
    embeddings2_norm = embeddings2 / embeddings2.norm(dim=-1, keepdim=True)
    
    similarities = (embeddings1_norm * embeddings2_norm).sum(dim=-1)
    
    return similarities.mean().item()


def format_metrics_report(metrics: Dict[str, float]) -> str:
    """
    Format metrics as a readable report.
    
    Args:
        metrics: Dictionary of metrics
    
    Returns:
        Formatted string
    """
    report = "Evaluation Metrics:\n"
    report += "=" * 50 + "\n"
    
    for metric_name, value in sorted(metrics.items()):
        report += f"{metric_name.upper():<15}: {value:.4f}\n"
    
    report += "=" * 50
    
    return report


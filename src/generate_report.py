from loguru import logger
from src.config import REPORT_PATH

def generate_report(classifiers, best_params, metrics, cross_val_scores):
    
    logger.info('generating report.md')
    with open(REPORT_PATH, 'w') as f:
        f.write('# Cancer Classification Report\n\n')
        
        f.write('## Evaluation Metrics\n\n')
        for name in metrics.keys():
            f.write(f'### {name}\n\n')
            for metric_name in metrics[name].keys():
                f.write(f'- {metric_name}: {metrics[name][metric_name]:.4f}\n')
            f.write('\n')
        
        f.write('## Cross-Validation Scores\n\n')
        for name in cross_val_scores.keys():
            f.write(f'- {name}: {cross_val_scores[name].mean():.4f} (+/- {cross_val_scores[name].std():.4f})\n')
        f.write('\n')
        
        f.write('## Classification Reports\n\n')
        for name in classifiers.keys():
            name_path = name.replace(' ', '_')
            # f.write(f'### {name}\n\n')
            # f.write(f'![Classification Report for {name}](reports/figures/classification_report_{name_path}.png)\n\n')
            
            f.write(f'### Best Params for {name} model\n\n')
            best_param = {**best_params[name]}
            f.write(f'- {best_param}\n\n')
            
            f.write('### ROC Curve\n\n')
            f.write(f'### {name}\n\n')
            
            f.write(f'![ROC Curve for {name}](reports/figures/roc_curve_{name_path}.png)\n\n')
        
            f.write('### Precision-Recall Curve\n\n')
            f.write(f'### {name}\n\n')
            f.write(f'![Precision-Recall Curve for {name}](reports/figures/precision_recall_curve_{name_path}.png)\n\n')
            
    return 
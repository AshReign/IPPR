import copy
from decimal import Decimal
import pytest
from p2.selection import THRESHOLDS, select

PROTOCOL = 'f' * 64


def fixtures():
    inventory = {'status': 'completed_formal_run', 'actual_epochs': 2, 'p1_commit': 'abc',
                 'dev_manifest_sha256': 'd' * 64,
                 'checkpoints': [{'sha256': str(epoch) * 64, 'completed_epoch': epoch} for epoch in (1, 2)]}
    scores = {'schema': 'p2-p5-dev-scores-v1', 'status': 'complete', 'split': 'dev',
              'selection_protocol_sha256': PROTOCOL, 'p1_commit': 'abc', 'metric_source': 'P5',
              'evaluator_source_sha256': 'a' * 64, 'output_adapter_sha256': 'b' * 64,
              'prediction_config_sha256': 'c' * 64, 'dev_manifest_sha256': 'd' * 64,
              'reuse_equivalence_report_sha256': 'e' * 64,
              'rows': [{'checkpoint_sha256': str(epoch) * 64, 'completed_epoch': epoch,
                        'threshold': str(threshold), 'status': 'complete',
                        'f2': '0.5', 'recall': '0.6', 'precision': '0.4'}
                       for epoch in (1, 2) for threshold in THRESHOLDS]}
    return inventory, scores


def test_exact_ties_choose_earlier_epoch_then_lower_threshold():
    inventory, scores = fixtures()
    result = select(scores, inventory, PROTOCOL)
    assert result['winner']['completed_epoch'] == 1
    assert Decimal(result['winner']['threshold']) == Decimal('0.1')
    assert result['scoring_combinations'] == 34
    assert result['status'].endswith('not_yet_frozen')


def test_full_precision_not_rounded_display():
    inventory, scores = fixtures()
    scores['rows'][-1]['f2'] = '0.50000000000000000001'
    assert select(scores, inventory, PROTOCOL)['winner'] == scores['rows'][-1]


def test_recall_then_precision_before_epoch():
    inventory, scores = fixtures()
    scores['rows'][-1]['recall'] = '0.61'
    assert select(scores, inventory, PROTOCOL)['winner'] == scores['rows'][-1]
    scores['rows'][-2]['recall'] = '0.61'
    scores['rows'][-2]['precision'] = '0.41'
    assert select(scores, inventory, PROTOCOL)['winner'] == scores['rows'][-2]


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'wrong_split', 'wrong_source', 'nan',
                                     'bad_epoch', 'off_grid', 'failed', 'wrong_manifest'])
def test_rejects_incomplete_or_untrusted_matrix(change):
    inventory, scores = fixtures()
    if change == 'missing': scores['rows'].pop()
    elif change == 'duplicate': scores['rows'].append(copy.deepcopy(scores['rows'][0]))
    elif change == 'wrong_split': scores['split'] = 'val'
    elif change == 'wrong_source': scores['metric_source'] = 'Ultralytics'
    elif change == 'nan': scores['rows'][0]['f2'] = 'NaN'
    elif change == 'bad_epoch': scores['rows'][0]['completed_epoch'] = 0
    elif change == 'off_grid': scores['rows'][0]['threshold'] = '0.12'
    elif change == 'failed': scores['rows'][0]['status'] = 'failed'
    elif change == 'wrong_manifest': scores['dev_manifest_sha256'] = '0' * 64
    with pytest.raises(ValueError): select(scores, inventory, PROTOCOL)

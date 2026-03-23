"""Tests para stats/sample_size.py — calculadora de tamano muestral."""


from stats.sample_size import (
    sample_size_anova,
    sample_size_correlation,
    sample_size_proportions,
    sample_size_survival,
    sample_size_ttest_ind,
    sample_size_ttest_paired,
    sensitivity_table,
)


class TestTTestInd:
    def test_medium_effect(self):
        r = sample_size_ttest_ind(0.5, alpha=0.05, power=0.8)
        assert r['success']
        assert r['n_per_group'] == 64
        assert r['n_total'] == 128

    def test_large_effect_needs_less_n(self):
        r_large = sample_size_ttest_ind(0.8, alpha=0.05, power=0.8)
        r_small = sample_size_ttest_ind(0.2, alpha=0.05, power=0.8)
        assert r_large['n_per_group'] < r_small['n_per_group']

    def test_higher_power_needs_more_n(self):
        r_80 = sample_size_ttest_ind(0.5, power=0.8)
        r_90 = sample_size_ttest_ind(0.5, power=0.9)
        assert r_90['n_per_group'] > r_80['n_per_group']

    def test_one_sided_needs_less_n(self):
        r_two = sample_size_ttest_ind(0.5, alternative='two-sided')
        r_one = sample_size_ttest_ind(0.5, alternative='larger')
        assert r_one['n_per_group'] < r_two['n_per_group']

    def test_zero_effect_fails(self):
        r = sample_size_ttest_ind(0)
        assert not r['success']

    def test_negative_effect_fails(self):
        r = sample_size_ttest_ind(-0.5)
        assert not r['success']

    def test_n_is_integer(self):
        r = sample_size_ttest_ind(0.5)
        assert isinstance(r['n_per_group'], int)
        assert isinstance(r['n_total'], int)

    def test_result_has_metadata(self):
        r = sample_size_ttest_ind(0.5, alpha=0.01, power=0.9)
        assert r['alpha'] == 0.01
        assert r['power'] == 0.9
        assert r['effect_size'] == 0.5
        assert r['test'] == 't-test independiente'


class TestTTestPaired:
    def test_medium_effect(self):
        r = sample_size_ttest_paired(0.5, alpha=0.05, power=0.8)
        assert r['success']
        assert r['n_per_group'] > 20

    def test_n_total_equals_n_pairs(self):
        r = sample_size_ttest_paired(0.5)
        assert r['n_total'] == r['n_per_group']

    def test_paired_needs_less_than_independent(self):
        r_paired = sample_size_ttest_paired(0.5)
        r_ind = sample_size_ttest_ind(0.5)
        assert r_paired['n_per_group'] < r_ind['n_per_group']

    def test_zero_effect_fails(self):
        r = sample_size_ttest_paired(0)
        assert not r['success']


class TestANOVA:
    def test_three_groups_medium(self):
        r = sample_size_anova(0.25, k_groups=3, alpha=0.05, power=0.8)
        assert r['success']
        assert r['n_total'] == r['n_per_group'] * 3

    def test_more_groups_needs_more_n_total(self):
        r3 = sample_size_anova(0.25, k_groups=3)
        r5 = sample_size_anova(0.25, k_groups=5)
        assert r5['n_total'] > r3['n_total']

    def test_one_group_fails(self):
        r = sample_size_anova(0.25, k_groups=1)
        assert not r['success']

    def test_zero_effect_fails(self):
        r = sample_size_anova(0, k_groups=3)
        assert not r['success']

    def test_result_has_k_groups(self):
        r = sample_size_anova(0.25, k_groups=4)
        assert r['k_groups'] == 4


class TestCorrelation:
    def test_medium_correlation(self):
        r = sample_size_correlation(0.3, alpha=0.05, power=0.8)
        assert r['success']
        assert r['n_total'] > 50

    def test_strong_correlation_needs_less_n(self):
        r_strong = sample_size_correlation(0.5)
        r_weak = sample_size_correlation(0.1)
        assert r_strong['n_total'] < r_weak['n_total']

    def test_negative_correlation_works(self):
        r_pos = sample_size_correlation(0.3)
        r_neg = sample_size_correlation(-0.3)
        assert r_pos['n_total'] == r_neg['n_total']

    def test_zero_correlation_fails(self):
        r = sample_size_correlation(0)
        assert not r['success']

    def test_r_equals_one_fails(self):
        r = sample_size_correlation(1.0)
        assert not r['success']

    def test_n_per_group_equals_n_total(self):
        r = sample_size_correlation(0.3)
        assert r['n_per_group'] == r['n_total']


class TestProportions:
    def test_basic_case(self):
        r = sample_size_proportions(0.3, 0.5, alpha=0.05, power=0.8)
        assert r['success']
        assert r['n_per_group'] > 0
        assert r['n_total'] == r['n_per_group'] * 2

    def test_larger_difference_needs_less_n(self):
        r_small = sample_size_proportions(0.4, 0.5)
        r_large = sample_size_proportions(0.2, 0.5)
        assert r_large['n_per_group'] < r_small['n_per_group']

    def test_equal_proportions_fails(self):
        r = sample_size_proportions(0.5, 0.5)
        assert not r['success']

    def test_invalid_proportion_fails(self):
        r = sample_size_proportions(0, 0.5)
        assert not r['success']
        r = sample_size_proportions(0.5, 1.0)
        assert not r['success']

    def test_result_has_effect_size_h(self):
        r = sample_size_proportions(0.3, 0.5)
        assert 'effect_size_h' in r
        assert r['effect_size_h'] > 0


class TestSurvival:
    def test_basic_case(self):
        r = sample_size_survival(0.5, alpha=0.05, power=0.8)
        assert r['success']
        assert r['n_events'] > 0

    def test_hr_close_to_one_needs_more_events(self):
        r_far = sample_size_survival(0.5)
        r_close = sample_size_survival(0.8)
        assert r_close['n_events'] > r_far['n_events']

    def test_hr_one_fails(self):
        r = sample_size_survival(1.0)
        assert not r['success']

    def test_hr_zero_fails(self):
        r = sample_size_survival(0)
        assert not r['success']

    def test_unequal_ratio(self):
        r = sample_size_survival(0.5, ratio=2.0)
        assert r['success']
        assert r['ratio'] == 2.0

    def test_negative_ratio_fails(self):
        r = sample_size_survival(0.5, ratio=-1)
        assert not r['success']

    def test_hr_greater_than_one_works(self):
        r = sample_size_survival(2.0)
        assert r['success']
        assert r['n_events'] > 0


class TestSensitivityTable:
    def test_returns_rows(self):
        rows = sensitivity_table(
            sample_size_ttest_ind,
            effect_sizes=[0.2, 0.5, 0.8],
        )
        assert len(rows) == 3

    def test_has_power_columns(self):
        rows = sensitivity_table(
            sample_size_ttest_ind,
            effect_sizes=[0.5],
            powers=(0.7, 0.8, 0.9),
        )
        assert 'power_70' in rows[0]
        assert 'power_80' in rows[0]
        assert 'power_90' in rows[0]

    def test_higher_power_higher_n(self):
        rows = sensitivity_table(
            sample_size_ttest_ind,
            effect_sizes=[0.5],
            powers=(0.7, 0.9),
        )
        assert rows[0]['power_90'] > rows[0]['power_70']

    def test_correlation_table(self):
        rows = sensitivity_table(
            sample_size_correlation,
            effect_sizes=[0.1, 0.3, 0.5],
        )
        assert len(rows) == 3
        assert rows[0]['power_80'] > rows[2]['power_80']

    def test_survival_table(self):
        rows = sensitivity_table(
            sample_size_survival,
            effect_sizes=[0.5, 0.7, 0.9],
        )
        assert len(rows) == 3

    def test_anova_table_with_k_groups(self):
        rows = sensitivity_table(
            sample_size_anova,
            effect_sizes=[0.1, 0.25, 0.4],
            k_groups=3,
        )
        assert len(rows) == 3

# HW 7, Problems 1, 2, 3, 4, 5, and 6
# REQUIRES THE MOST RECENT Labs_SurvivalModel-master and HPM473TO BE LOADED IN CONTENT ROOT

# PROBLEM 1
# Code copied and modified from RunSteadyState.py
# Version used downloaded on March 24th
import SurvivalModelClasses as SurvivalCls

MORTALITY_PROB = 0.1    # annual probability of mortality (left as default from GitHub code)
TIME_STEPS = 1000       # simulation length (1000 years)
SIM_POP_SIZE = 573     # population size of the simulated cohort (changed to 573 per Problem 1)
ALPHA = 0.05            # significance level

# create a cohort of patients
myCohort = SurvivalCls.Cohort(id=1, pop_size=SIM_POP_SIZE, mortality_prob=MORTALITY_PROB)

# simulate the cohort
cohortOutcome = myCohort.simulate(TIME_STEPS)

# Print the percent of patients survived beyond 5 years
survivors = cohortOutcome.get_survival_times()
print("Problem 1: Percent of patients who survived beyond 5 years:",((sum(i > 5 for i in survivors))/573)*100, "%",'\n')

# PROBLEM 2
# Print out response
print("Problem 2: The scenario described should follow a binomial distribution with N trials (number of patients of cohort) and probability of success q (5-year survival probability)",'\n')

# PROBLEM 3
# Import binomial distribution from scipy
from scipy.stats import binom

# Calculate the likelihood that a clinical study reports 400 of 573 participants survived at the end of the 5-year study period if 50% of the patients in our simulated cohort survived beyond 5 years using binomial distribution
print("Problem 3: Likelihood of given scenario:", binom.pmf(k = 400, n = 573, p = 0.5), '\n')

# PROBLEM 4
# Code adapted from CalibrationSettings.py and CalibrationClasses.py
# Import full scipy package
from enum import Enum
import numpy as np
import scr.InOutFunctions as InOutSupport
import scr.StatisticalClasses as StatSupport
import scr.FormatFunctions as FormatSupport


# Set up calibration
SIM_POP_SIZE = 1000       # population size of simulated cohorts
TIME_STEPS = 1000        # length of simulation
ALPHA = 0.05             # significance level for calculating confidence intervals
NUM_SIM_COHORTS = 500   # number of simulated cohorts used to calculate prediction intervals

# Clinical trail info
OBS_N = 573        # number of patients involved in the study
OBS_SUCCESS = 400    # observed number of patients who survived 5-years
OBS_PROB = (400/573)      # observed 5-year survival probability

# how to sample the posterior distribution of mortality probability: completely unknown true annual mortality
# minimum, maximum and the number of samples for the mortality probability
POST_L, POST_U, POST_N = 0, 1, 1000

class CalibrationColIndex(Enum):
    """ indices of columns in the calibration results cvs file  """
    ID = 0          # cohort ID
    W = 1  # likelihood weight
    MORT_PROB = 2   # mortality probability


class Calibration:
    def __init__(self):
        """ initializes the calibration object"""
        np.random.seed(1)   # specifying the seed of the numpy random number generator
        self._cohortIDs = range(POST_N)   # IDs of cohorts to simulate
        self._mortalitySamples = []      # values of mortality probability at which the posterior should be sampled
        self._mortalityResamples = []    # resampled values for constructing posterior estimate and interval
        self._weights = []               # likelihood weights of sampled mortality probabilities
        self._normalizedWeights = []     # normalized likelihood weights (sums to 1)
        self._csvRows = \
            [['Cohort ID', 'Likelihood Weights' ,'Mortality Prob']]  # list containing the calibration results

    def sample_posterior(self):
        """ sample the posterior distribution of the mortality probability """

        # find values of mortality probability at which the posterior should be evaluated
        self._mortalitySamples = np.random.uniform(
            low=POST_L,
            high=POST_U,
            size=POST_N)

        # create a multi cohort
        multiCohort = SurvivalCls.MultiCohort(
            ids=self._cohortIDs,
            mortality_probs=self._mortalitySamples,
            pop_sizes=[SIM_POP_SIZE]*POST_N
        )

        # simulate the multi cohort
        multiCohort.simulate(TIME_STEPS)

        # calculate the likelihood of each simulated cohort
        for cohort_id in self._cohortIDs:

            # get the 5-year survival percentage of the cohort
            survivaltimelist = multiCohort._survivalTimes[cohort_id]
            fiveyearsurvival = sum(j > 5 for j in survivaltimelist)/1000

            # construct a binomial distribution
            # with number of successes (5-year survival) and 5-year survival rate estimate from clinical trial
            # evaluate this pdf (probability density function) at the mean reported in the clinical study.
            weight = binom.pmf(
                k=OBS_SUCCESS,
                n=OBS_N,
                p=fiveyearsurvival,
                loc=0)

            # store the weight
            self._weights.append(weight)

        # normalize the likelihood weights
        sum_weights = np.sum(self._weights)
        self._normalizedWeights = np.divide(self._weights, sum_weights)

        # re-sample mortality probability (with replacement) according to likelihood weights
        self._mortalityResamples = np.random.choice(
            a=self._mortalitySamples,
            size=NUM_SIM_COHORTS,
            replace=True,
            p=self._normalizedWeights)

        # produce the list to report the results
        for i in range(0, len(self._mortalitySamples)):
            self._csvRows.append(
                [self._cohortIDs[i], self._normalizedWeights[i], self._mortalitySamples[i]])

        # write the calibration result into a csv file
        InOutSupport.write_csv('CalibrationResults.csv', self._csvRows)

    def get_mortality_resamples(self):
        """
        :return: mortality resamples
        """
        return self._mortalityResamples

    def get_mortality_estimate_credible_interval(self, alpha, deci):
        """
        :param alpha: the significance level
        :param deci: decimal places
        :returns text in the form of 'mean (lower, upper)' of the posterior distribution"""

        # calculate the credible interval
        sum_stat = StatSupport.SummaryStat('Posterior samples', self._mortalityResamples)

        estimate = sum_stat.get_mean()  # estimated mortality probability
        credible_interval = sum_stat.get_PI(alpha)  # credible interval

        return FormatSupport.format_estimate_interval(estimate, credible_interval, deci)

    def get_effective_sample_size(self):
        """
        :returns: the effective sample size
        """
        return 1 / np.sum(self._normalizedWeights ** 2)

# Run the calibration, code adapted from RunCalibration.py
# create a calibration object
calibration = Calibration()

# sample the posterior of the mortality probability
calibration.sample_posterior()

# Estimate of mortality probability and the posterior interval
print('Problem 4: Estimate of mortality probability ({:.{prec}%} credible interval):'.format(1-ALPHA, prec=0),
      calibration.get_mortality_estimate_credible_interval(ALPHA, 4), "\n")

# PROBLEM 5
# Code copied from CalibrationClasses.py
class CalibratedModel:
    """ to run the calibrated survival model """

    def __init__(self, cvs_file_name, drug_effectiveness_ratio=1):
        """ extracts seeds, mortality probabilities and the associated likelihood from
        the csv file where the calibration results are stored
        :param cvs_file_name: name of the csv file where the calibrated results are stored
        :param calibrated_model_with_drug: calibrated model simulated when drug is available
        """

        # read the columns of the csv files containing the calibration results
        cols = InOutSupport.read_csv_cols(
            file_name=cvs_file_name,
            n_cols=3,
            if_ignore_first_row=True,
            if_convert_float=True)

        # store likelihood weights, cohort IDs and sampled mortality probabilities
        self._cohortIDs = cols[CalibrationColIndex.ID.value].astype(int)
        self._weights = cols[CalibrationColIndex.W.value]
        self._mortalityProbs = cols[CalibrationColIndex.MORT_PROB.value] * drug_effectiveness_ratio
        self._multiCohorts = None  # multi-cohort

    def simulate(self, num_of_simulated_cohorts, cohort_size, time_steps, cohort_ids=None):
        """ simulate the specified number of cohorts based on their associated likelihood weight
        :param num_of_simulated_cohorts: number of cohorts to simulate
        :param cohort_size: the population size of cohorts
        :param time_steps: simulation length
        :param cohort_ids: ids of cohort to simulate
        """
        # resample cohort IDs and mortality probabilities based on their likelihood weights
        # sample (with replacement) from indices [0, 1, 2, ..., number of weights] based on the likelihood weights
        sampled_row_indices = np.random.choice(
            a=range(0, len(self._weights)),
            size=num_of_simulated_cohorts,
            replace=True,
            p=self._weights)

        # use the sampled indices to populate the list of cohort IDs and mortality probabilities
        resampled_ids = []
        resampled_probs = []
        for i in sampled_row_indices:
            resampled_ids.append(self._cohortIDs[i])
            resampled_probs.append(self._mortalityProbs[i])

        # simulate the desired number of cohorts
        if cohort_ids is None:
            # if cohort ids are not provided, use the ids stored in the calibration results
            self._multiCohorts = SurvivalCls.MultiCohort(
                ids=resampled_ids,
                pop_sizes=[cohort_size] * num_of_simulated_cohorts,
                mortality_probs=resampled_probs)
        else:
            # if cohort ids are provided, use them instead of the ids stored in the calibration results
            self._multiCohorts = SurvivalCls.MultiCohort(
                ids=cohort_ids,
                pop_sizes=[cohort_size] * num_of_simulated_cohorts,
                mortality_probs=resampled_probs)

        # simulate all cohorts
        self._multiCohorts.simulate(time_steps)

    def get_all_mean_survival(self):
        """ :returns a list of mean survival time for all simulated cohorts"""
        return self._multiCohorts.get_all_mean_survival()

    def get_mean_survival_time_proj_interval(self, alpha, deci):
        """
        :param alpha: the significance level
        :param deci: decimal places
        :returns text in the form of 'mean (lower, upper)' of projection interval
        """

        mean = self._multiCohorts.get_overall_mean_survival()
        proj_interval = self._multiCohorts.get_PI_mean_survival(alpha)

        return FormatSupport.format_estimate_interval(mean, proj_interval, deci)

# Run the calibrated model, code adapted from RunCalibration.py

# initialize a calibrated model
calibrated_model = CalibratedModel('CalibrationResults.csv')

# simulate the calibrated model
calibrated_model.simulate(NUM_SIM_COHORTS, SIM_POP_SIZE, TIME_STEPS)

# report mean and projection interval
print('Problem 5: Mean survival time and {:.{prec}%} projection interval:'.format(1 - ALPHA, prec=0),
      calibrated_model.get_mean_survival_time_proj_interval(ALPHA, deci=4), "\n")

# PROBLEM 6
# All code modified from same sources as mentioned in Problem 5

# New clinical trail info
OBS_N_2 = 1146        # number of patients involved in the study
OBS_SUCCESS_2 = 800    # observed number of patients who survived 5-years
OBS_PROB_2 = (800/1146)      # observed 5-year survival probability

class CalibrationColIndex(Enum):
    """ indices of columns in the calibration results cvs file  """
    ID = 0          # cohort ID
    W = 1  # likelihood weight
    MORT_PROB = 2   # mortality probability


class Calibration:
    def __init__(self):
        """ initializes the calibration object"""
        np.random.seed(1)   # specifying the seed of the numpy random number generator
        self._cohortIDs = range(POST_N)   # IDs of cohorts to simulate
        self._mortalitySamples = []      # values of mortality probability at which the posterior should be sampled
        self._mortalityResamples = []    # resampled values for constructing posterior estimate and interval
        self._weights = []               # likelihood weights of sampled mortality probabilities
        self._normalizedWeights = []     # normalized likelihood weights (sums to 1)
        self._csvRows = \
            [['Cohort ID', 'Likelihood Weights' ,'Mortality Prob']]  # list containing the calibration results

    def sample_posterior(self):
        """ sample the posterior distribution of the mortality probability """

        # find values of mortality probability at which the posterior should be evaluated
        self._mortalitySamples = np.random.uniform(
            low=POST_L,
            high=POST_U,
            size=POST_N)

        # create a multi cohort
        multiCohort = SurvivalCls.MultiCohort(
            ids=self._cohortIDs,
            mortality_probs=self._mortalitySamples,
            pop_sizes=[SIM_POP_SIZE]*POST_N
        )

        # simulate the multi cohort
        multiCohort.simulate(TIME_STEPS)

        # calculate the likelihood of each simulated cohort
        for cohort_id in self._cohortIDs:

            # get the 5-year survival percentage of the cohort
            survivaltimelist = multiCohort._survivalTimes[cohort_id]
            fiveyearsurvival = sum(j > 5 for j in survivaltimelist)/1000

            # construct a binomial distribution
            # with number of successes (5-year survival) and 5-year survival rate estimate from clinical trial
            # evaluate this pdf (probability density function) at the mean reported in the clinical study.
            weight = binom.pmf(
                k=OBS_SUCCESS_2,
                n=OBS_N_2,
                p=fiveyearsurvival,
                loc=0)

            # store the weight
            self._weights.append(weight)

        # normalize the likelihood weights
        sum_weights = np.sum(self._weights)
        self._normalizedWeights = np.divide(self._weights, sum_weights)

        # re-sample mortality probability (with replacement) according to likelihood weights
        self._mortalityResamples = np.random.choice(
            a=self._mortalitySamples,
            size=NUM_SIM_COHORTS,
            replace=True,
            p=self._normalizedWeights)

        # produce the list to report the results
        for i in range(0, len(self._mortalitySamples)):
            self._csvRows.append(
                [self._cohortIDs[i], self._normalizedWeights[i], self._mortalitySamples[i]])

        # write the calibration result into a csv file
        InOutSupport.write_csv('CalibrationResults_2.csv', self._csvRows)

    def get_mortality_resamples(self):
        """
        :return: mortality resamples
        """
        return self._mortalityResamples

    def get_mortality_estimate_credible_interval(self, alpha, deci):
        """
        :param alpha: the significance level
        :param deci: decimal places
        :returns text in the form of 'mean (lower, upper)' of the posterior distribution"""

        # calculate the credible interval
        sum_stat = StatSupport.SummaryStat('Posterior samples', self._mortalityResamples)

        estimate = sum_stat.get_mean()  # estimated mortality probability
        credible_interval = sum_stat.get_PI(alpha)  # credible interval

        return FormatSupport.format_estimate_interval(estimate, credible_interval, deci)

    def get_effective_sample_size(self):
        """
        :returns: the effective sample size
        """
        return 1 / np.sum(self._normalizedWeights ** 2)

# Run the calibration, code adapted from RunCalibration.py
# create a calibration object
calibration = Calibration()

# sample the posterior of the mortality probability
calibration.sample_posterior()

# Estimate of mortality probability and the posterior interval
print('Problem 6: Estimate of new mortality probability ({:.{prec}%} credible interval):'.format(1-ALPHA, prec=0),
      calibration.get_mortality_estimate_credible_interval(ALPHA, 4))

# Code copied from CalibrationClasses.py
class CalibratedModel:
    """ to run the calibrated survival model """

    def __init__(self, cvs_file_name, drug_effectiveness_ratio=1):
        """ extracts seeds, mortality probabilities and the associated likelihood from
        the csv file where the calibration results are stored
        :param cvs_file_name: name of the csv file where the calibrated results are stored
        :param calibrated_model_with_drug: calibrated model simulated when drug is available
        """

        # read the columns of the csv files containing the calibration results
        cols = InOutSupport.read_csv_cols(
            file_name=cvs_file_name,
            n_cols=3,
            if_ignore_first_row=True,
            if_convert_float=True)

        # store likelihood weights, cohort IDs and sampled mortality probabilities
        self._cohortIDs = cols[CalibrationColIndex.ID.value].astype(int)
        self._weights = cols[CalibrationColIndex.W.value]
        self._mortalityProbs = cols[CalibrationColIndex.MORT_PROB.value] * drug_effectiveness_ratio
        self._multiCohorts = None  # multi-cohort

    def simulate(self, num_of_simulated_cohorts, cohort_size, time_steps, cohort_ids=None):
        """ simulate the specified number of cohorts based on their associated likelihood weight
        :param num_of_simulated_cohorts: number of cohorts to simulate
        :param cohort_size: the population size of cohorts
        :param time_steps: simulation length
        :param cohort_ids: ids of cohort to simulate
        """
        # resample cohort IDs and mortality probabilities based on their likelihood weights
        # sample (with replacement) from indices [0, 1, 2, ..., number of weights] based on the likelihood weights
        sampled_row_indices = np.random.choice(
            a=range(0, len(self._weights)),
            size=num_of_simulated_cohorts,
            replace=True,
            p=self._weights)

        # use the sampled indices to populate the list of cohort IDs and mortality probabilities
        resampled_ids = []
        resampled_probs = []
        for i in sampled_row_indices:
            resampled_ids.append(self._cohortIDs[i])
            resampled_probs.append(self._mortalityProbs[i])

        # simulate the desired number of cohorts
        if cohort_ids is None:
            # if cohort ids are not provided, use the ids stored in the calibration results
            self._multiCohorts = SurvivalCls.MultiCohort(
                ids=resampled_ids,
                pop_sizes=[cohort_size] * num_of_simulated_cohorts,
                mortality_probs=resampled_probs)
        else:
            # if cohort ids are provided, use them instead of the ids stored in the calibration results
            self._multiCohorts = SurvivalCls.MultiCohort(
                ids=cohort_ids,
                pop_sizes=[cohort_size] * num_of_simulated_cohorts,
                mortality_probs=resampled_probs)

        # simulate all cohorts
        self._multiCohorts.simulate(time_steps)

    def get_all_mean_survival(self):
        """ :returns a list of mean survival time for all simulated cohorts"""
        return self._multiCohorts.get_all_mean_survival()

    def get_mean_survival_time_proj_interval(self, alpha, deci):
        """
        :param alpha: the significance level
        :param deci: decimal places
        :returns text in the form of 'mean (lower, upper)' of projection interval
        """

        mean = self._multiCohorts.get_overall_mean_survival()
        proj_interval = self._multiCohorts.get_PI_mean_survival(alpha)

        return FormatSupport.format_estimate_interval(mean, proj_interval, deci)

# Run the calibrated model, code adapted from RunCalibration.py

# initialize a calibrated model
calibrated_model = CalibratedModel('CalibrationResults_2.csv')

# simulate the calibrated model
calibrated_model.simulate(NUM_SIM_COHORTS, SIM_POP_SIZE, TIME_STEPS)

# report mean and projection interval
print('New mean survival time and {:.{prec}%} projection interval:'.format(1 - ALPHA, prec=0),
      calibrated_model.get_mean_survival_time_proj_interval(ALPHA, deci=4))
print("Both the credible interval of the estimated annual mortality probability and the projection interval of the mean survival time are smaller, indicating more precise estimates.")







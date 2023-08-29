import numpy as np
from scipy.stats import bernoulli
from scipy.stats import poisson
import random as rand
import matplotlib.pyplot as plt
import scipy as sp

# constants for organizing drugs of different classes
CLASS_A = 0  # Class A represents Disease Modifying Therapies (DMT) drugs used for MS
CLASS_B = 1  # Class B represents Immune Modulators drugs (i.e. B cell immune suppressants like Ocrelizumab)
CLASS_C = 2  # Class C represents a no-drug baseline for a person with MS

# Risk and Probabilities Taken From Research
PROB_OF_CONTRACTION = 0.026  # See Source 2
PROB_OF_HOSPITALIZATION = [0.133, 0.251, 0.372]  # See Source 3
PROB_OF_ICU = [0.194, 0.319, 0.175]  # See Source 3
CO_CONTACT_RATE = 0.67  # See Source 1
PROB_OF_BREAKTHROUGH = 0.06  # See Source 4
IC_INCREASED_BREAKTHROUGH = 3  # See Source 4
RELAPSE_RISK = [0.33, 0.15, 0.5] # See Sources 5,6,7

# Rejection sampling consts
N_SAMPLES = 100000
CONTACT = 0
COVID = 1
HOSPITAL = 2
ICU = 3
CLASS = -1

# Makes a random sample of an MS patient following the Bayesian Network (see write-up)
def make_sample(prob_contact=CO_CONTACT_RATE, increased_risk_of_contraction=2):
    drug_class = rand.randrange(0, 3)

    contact = bernoulli.rvs(prob_contact)  # Comes into contact with covid positive person
    if (not contact):
        return [0, 0, 0, 0, drug_class]

    contract = bernoulli.rvs(
        increased_risk_of_contraction * PROB_OF_CONTRACTION) if drug_class == CLASS_B else bernoulli.rvs(
        PROB_OF_CONTRACTION)  # contracts covid-19
    if (not contract):
        return [1, 0, 0, 0, drug_class]

    hospitalization = bernoulli.rvs(PROB_OF_HOSPITALIZATION[drug_class])  # requires hospitalization
    if (not hospitalization):
        return [1, 1, 0, 0, drug_class]

    icu = bernoulli.rvs(PROB_OF_ICU[drug_class])  # requires ICU care
    if (not icu):
        return [1, 1, 1, 0, drug_class]

    return [1, 1, 1, 1, drug_class]

# Makes vaccinated sample based on the second iteration of the Bayesian Network
# Filters out samples based on whether case was a breakthrough
def make_vaccinated_sample():
    sample = make_sample()
    if (sample[CONTACT] and sample[COVID]):
        breakthrough = bernoulli.rvs(IC_INCREASED_BREAKTHROUGH * PROB_OF_BREAKTHROUGH) \
            if sample[CLASS] == CLASS_B else bernoulli.rvs(PROB_OF_BREAKTHROUGH)
        if (not breakthrough):
            return [1, 0, 0, 0, sample[CLASS]]

    return sample


# Calculates the likelihood that an MS patient was on a specific treatment
# given that they had a severe covid case
def drug_given_severe_case(is_vaccinated=False):
    icu_tallies = np.array([0, 0, 0])
    hospital_tallies = np.array([0, 0, 0])
    icu_total = 0
    hospital_total = 0

    for i in range(0, N_SAMPLES):
        sample = make_sample() if (not is_vaccinated) else make_vaccinated_sample()
        if (not sample[HOSPITAL]):
            continue
        hospital_tallies[sample[CLASS]] += 1
        hospital_total += 1
        if (sample[ICU]):
            icu_tallies[sample[CLASS]] += 1
            icu_total += 1

    return hospital_tallies / hospital_total, icu_tallies / icu_total

# Calculates the likelihood that the MS patient will have a severe covid case
# given that they are on a specific treatment
def severe_case_given_drug(is_vaccinated=False):
    icu_tallies = np.array([0, 0, 0])
    hospital_tallies = np.array([0, 0, 0])
    drug_totals = np.array([0, 0, 0])

    for i in range(0, N_SAMPLES):
        sample = make_sample() if (not is_vaccinated) else make_vaccinated_sample()
        drug_totals[sample[CLASS]] += 1
        if (not sample[HOSPITAL]):
            continue
        hospital_tallies[sample[CLASS]] += 1
        if (sample[ICU]):
            icu_tallies[sample[CLASS]] += 1

    print("A) Hospital:", hospital_tallies[CLASS_A] / drug_totals[CLASS_A], "ICU:", icu_tallies[CLASS_A] / drug_totals[CLASS_A])
    print("B) Hospital:", hospital_tallies[CLASS_B] / drug_totals[CLASS_B], "ICU:", icu_tallies[CLASS_B] / drug_totals[CLASS_B])
    print("C) Hospital:", hospital_tallies[CLASS_C] / drug_totals[CLASS_C], "ICU:", icu_tallies[CLASS_C] / drug_totals[CLASS_C])
    return [hospital_tallies[CLASS_A] / drug_totals[CLASS_A], hospital_tallies[CLASS_B] / drug_totals[CLASS_B], hospital_tallies[CLASS_C] / drug_totals[CLASS_C]], [icu_tallies[CLASS_A] / drug_totals[CLASS_A], icu_tallies[CLASS_B] / drug_totals[CLASS_B], icu_tallies[CLASS_C] / drug_totals[CLASS_C]]

# Graphs breakdown of severe cases based on drug class
def graph_severe_case_distributions():
    names = ["Class A", "Class B", "Class C"]
    hospitalization, icu = drug_given_severe_case()
    vac_hospital, vac_icu = drug_given_severe_case(is_vaccinated=True)
    plt.figure(figsize=(8, 8))
    plt.suptitle("Breakdown of Severe Cases of Covid in Patients with MS")

    plt.subplot(2, 2, 1)
    plt.bar(names, hospitalization)
    plt.title("Hospitalizations")
    plt.subplot(2, 2, 2)
    plt.title("ICU Admissions")
    plt.bar(names, icu)
    plt.subplot(2, 2, 3)
    plt.title("Hospitalizations Post-Vaccination")
    plt.bar(names, vac_hospital)
    plt.subplot(2, 2, 4)
    plt.title("ICU Admissions Post-Vaccination")
    plt.bar(names, vac_icu)
    plt.show()

# Graphs poisson pmfs for likelihood of having severe case of covid as well as relapse
def risk_over_year (drug_class, num_exposures=52):
    hospital, icu = severe_case_given_drug()
    hospital_vax, icu_vax = severe_case_given_drug(is_vaccinated=True)
    hospital_lambda = hospital[drug_class]*num_exposures
    icu_lambda = icu[drug_class]*num_exposures
    h_vax_lambda = hospital_vax[drug_class]*num_exposures
    i_vax_lambda = icu_vax[drug_class]*num_exposures
    relapse_lambda = RELAPSE_RISK[drug_class]
    fig, ax = plt.subplots(1, 1)
    plt.suptitle("Yearly Risk of Severe Case or Relapse for Class " + chr(65 + drug_class))
    ax.set_xticks([0, 1, 2, 3])
    x2 = np.arange(poisson.ppf(0.0001, relapse_lambda),
                   poisson.ppf(0.9999, relapse_lambda))
    ax.plot(x2, poisson.pmf(x2, relapse_lambda), 'ko--', ms=8, label='# of Relapses', alpha=0.7)
    x1 = np.arange(poisson.ppf(0.0001, hospital_lambda),
                  poisson.ppf(0.9999, hospital_lambda))
    ax.plot(x1, poisson.pmf(x1, hospital_lambda), 'bo--', ms=8, label='# of Hospitalizations ', alpha=0.5)
    x3 = np.arange(poisson.ppf(0.0001, icu_lambda),
                   poisson.ppf(0.9999, icu_lambda))
    ax.plot(x3, poisson.pmf(x3, icu_lambda), 'ro--', ms=8, label='# of ICU Admissions', alpha=0.5)
    x4 = np.arange(poisson.ppf(0.0001, h_vax_lambda),
                   poisson.ppf(0.9999, h_vax_lambda))
    ax.plot(x4, poisson.pmf(x4, h_vax_lambda), 'go--', ms=8, label='# of Hospitalizations w Vaccine', alpha=0.5)
    x5 = np.arange(poisson.ppf(0.0001, i_vax_lambda),
                   poisson.ppf(0.9999, i_vax_lambda))
    ax.plot(x5, poisson.pmf(x5, i_vax_lambda), 'yo--', ms=8, label='# of ICU Admissions w Vaccine', alpha=0.5)
    ax.legend(loc='best', frameon=False)
    plt.show()

# Outputs Poisson probabilities for 0 or 1 occurences of severe case or recap
def poisson_dump(drug_class, num_exposures=52):
    hospital, icu = severe_case_given_drug()
    hospital_vax, icu_vax = severe_case_given_drug(is_vaccinated=True)
    hospital_lambda = hospital[drug_class] * num_exposures
    icu_lambda = icu[drug_class] * num_exposures
    h_vax_lambda = hospital_vax[drug_class] * num_exposures
    i_vax_lambda = icu_vax[drug_class] * num_exposures
    relapse_lambda = RELAPSE_RISK[drug_class]
    #print("relapses (0,1):", poisson.pmf(0, relapse_lambda), poisson.pmf(1, relapse_lambda))
    #print("hospital no vax", poisson.pmf(0, hospital_lambda),poisson.pmf(1, hospital_lambda))
    #print("icu no vax", poisson.pmf(0, icu_lambda), poisson.pmf(1, icu_lambda))
    #print("hospital vax", poisson.pmf(0, h_vax_lambda), poisson.pmf(1, h_vax_lambda))
    print("icu vax", poisson.pmf(0, i_vax_lambda), poisson.pmf(1, i_vax_lambda))


# Code to Run for Demo
severe_case_given_drug()
print("\n Now with Vaccines! \n")
severe_case_given_drug(is_vaccinated=True)
print("Graph time!")
graph_severe_case_distributions()
risk_over_year(drug_class=CLASS_B, num_exposures=52)

#============================================================================================================================================
# [_Class_R_B_ZC_Insur.py]
# [C:\Dropbox\_\work\Fin_Deriv\R_B_ZC_Insur\_Class_R_B_ZC_Insur.py]
#============================================================================================================================================
import streamlit as st
import datetime

class _Class_R_B_ZC_Insur:

    asset_class = 'R'
    security_type = 'B_ZC'
    ticker = asset_class + '_' + security_type + '_' + 'Insur'
    n = 1
    s_n = ''

    s_display_formula = (
                r'\Large'
                r'\begin{array}{lcl} '
                r'\\\\'
                r'     &R\_B\_ZC(0,t,T) &= &D(0,t)                          &\cdot &CF\_per \\'
                r'     &                &= &\left(\frac{1}{1+r\_per}\right) &\cdot &CF\_per \\'
                r'\\\\'
                r'\end{array}'
            )

    s_text_fields = (
        r'\begin{array}{lcl} '
        r'n [y] &= &\text{Time Horizon} \\'
        r'      &  &\text{(in years)} \\'
        r'\\\\'
        r'm [1/y] &= &\text{Coupon Frequency} \\'
        r'        &  &\text{(per year)} \\'
        r'\\\\'
        r'nm [1] &= &\text{Total Num Coupons} \\'
        r'       &  &\text{(periods)} \\'
        r'\\\\'
        r'per [y] &= &\text{Period between Coupons} \\'
        r'        &  &\text{(years)} \\'
        r'\\\\\\'
        r'r [\%] &= &\text{risk-free rate} \\'
        r'       &  &\text{(per year)} \\'
        r'\\'
        r'r\_per [\%] &= &\text{risk-free rate} \\'
        r'       &  &\text{(per period)} \\'
        r'\\\\\\\\'
        r'c [\%] &= &\text{coupon rate} \\'
        r'       &  &\text{(per year)} \\'
        r'\\'
        r'c\_per [\%] &= &\text{coupon rate} \\'
        r'       &  &\text{(per period)} \\'
        r'\\\\'
        r'N [SGD] &= &\text{Notional at t=0} \\'
        r'       &  &\text{(in SGD)} \\'
        r'\\\\'
        # r'CF [SGD/y] &= &\text{Cashflow} \\'
        # r'           &  &\text{(in SGD per year)} \\'
        r'\\'
        r'CF\_per [SGD/per] &= &\text{Cashflow} \\'
        r'                  &  &\text{(in SGD per period)} \\'
        r'\\\\'
        r'R\_B\_ZC(0,0,\infty) [SGD] &= &\text{Present Value} \\'
        r'                           &  &\text{of Zero Coupon Bond} \\'
        r'                           &  &\text{(in SGD)} \\'
        r'\\\\'
        r'Commission [SGD] &= &\text{Commission} \\'
        r'                 &  &\text{(in SGD)} \\'
        r'\\\\'
        r'\end{array}'
    )

    # Factsheet
    dt_launch = datetime.date(2024, 3, 6)
    i_policy_term_in_years = 3
    i_time_horizon_years_min = i_policy_term_in_years
    i_time_horizon_years_max = i_policy_term_in_years
    s_return_guaranteed = "Yes"
    s_issuer = "Life Insurance Corporation (Singapore) Pte Ltd"
    s_issuer_contact_number = "+6562234797"
    s_issuer_contact_email = "sales@licsingapore.com"
    s_product_plan = "Wealth Plus 3.1"
    s_premium_type = "Single Premium"
    s_currency = "SGD"
    f_premium_SGD_min = 20000
    f_premium_SGD_max = 200000
    f_premium_SGD_step = 5000
    s_par = "Non-Participating"
    s_product_desc = ("Wealth Plus 3.1 is a Single Premium non-participating endowment plan that gives  \n"
                      "a lump sum payout with guaranteed return of  \n"
                      "_3.62% annual return over 3 years.  \n"
                      "10.86% total return after 3 years.  \n"
                      )
    # Suitability
    s_eligibility_age_min = 18
    s_eligibility_age_max = 70
    s_eligibility_residency_SG_Citizen = "Yes"
    s_eligibility_residency_SG_PR = "Yes"
    s_eligibility_residency_SG_EP = "Yes"

    s_product_benefit_maturity = (
        "If the insured survives at the end of the policy term and this policy has not ended,  \n"
        "we will pay the guaranteed maturity benefit at the end of the policy term.  \n"
        "This policy will end when we make this payment.")
    s_product_benefit_death = "In the event of death of the life assured, the single premium is returned with interest, using the guaranteed simple interest rate."
    s_product_benefit_death_accidental = ("In the event of accidental death in the first year of the policy,  \n"
                                          "subject to the policyholder being under age 70,  \n"
                                          "an additional 10% of the single premium will be payable.")
    s_product_benefit_TPD = (
        "Upon diagnosis of TPD before age 65, the single premium is returned with interest, using the guaranteed simple interest rate.")
    s_product_benefit_exclusions = ("Death due to suicide within one year from the date of issue of Policy."
                                    "Back Dating is not allowed in the policy."
                                    "The Permanent Disability Benefit will not be paid if the disability has occurred as a result of Intentional acts (sane or insane) such as self -harm or attempted suicide."
                                    "i.  Criminal acts, war (decided or not), terrorism and chemical warfare."
                                    "ii. Participating in aviation (except as fare paying passenger or member crew of a commercial airline), "
                                    "any dangerous or hazardous sport or hobby such as (but not limited to) "
                                    "steeple chasing, polo, horse racing, underwater diving, hunting, motor vehicular racing, mountaineering or potholing "
                                    "or aerial sports such as skydiving, parachuting, bungee jumping."
                                    "iii. Any pre-existing conditions.")

    s_payment_payable_to = "Life Insurance Corporation (Singapore) Pte Ltd"
    s_payment_method_crossedcheque = "Yes"
    s_payment_method_cashiersorder = "Yes"
    s_payment_method_cashiersorder_bankerpayslipcopy = "Yes"
    s_payment_method_cashiersorder_purchaser_nric = "Yes"
    s_payment_method_interbanktransfer = "Yes"

    # * Applications will be processed on the system only on receipt of payment along with application.
    # Note: Life Insurance Corporation (Singapore) Pte Ltd accepts insurance premium payments from the proposer or from the legal spouse, parent or grandparent of the proposer only.
    # Underwriting
    # The policy is a guaranteed acceptance policy.
    # Requirements for AML /CFT compliance.
    s_medical_exam_needed = "No"
    s_back_dating_allowed = "No"
    s_riders = "No"
    s_beneficiary_assignment = "Yes"
    s_beneficiary_nomination = "Yes"

    s_surrender_allowed = "Yes"
    s_surrender_allowed_predeterminedfactors = "Yes"

    i_days_freelook = 14

    # Gross Commission to FA firms = % of the single premium including GST
    f_comm_pct_prem_plus_gst_to_FA = 0.25


    def __init__(self, asset_class, security_type, ticker):
        self.asset_class = asset_class
        self.security_type = security_type
        self.ticker = ticker

    # dunder method
    # whenever object is printed, it will print address
    # with this __str__ dunder method, it will print the return string
    def __str__(self):
        s = '['
        s = s + f'asset_class = {self.asset_class}' + ', '
        s = s + f'security_type = {self.security_type}' + ', '
        s = s + f'ticker = {self.ticker}' + ', '
        s = s + ']'
        return s






#============================================================================================================================================
# main
#============================================================================================================================================
if __name__ == "__main__":
    pass

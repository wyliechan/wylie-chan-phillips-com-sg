# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import numpy as np

import streamlit as st
from streamlit.hello.utils import show_code
import datetime
import sys

st.set_page_config(page_title="R_B_ZC_Insur", page_icon="📹")
st.markdown("# R_B_ZC_Insur")

# st.sidebar.header("R_B_ZC_Insur")
st.write(
    """ LIC
    """
)

class R_B_ZC_Insur:

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









# Create the Object using Standard Parameters
s_ticker = r"R_B_ZC_Insur___NKSgd=020-200_Prem1_T=3y_r=3.62\%_Endow_Par0_LIC_WealthPlus3.1"
R_B_ZC_Insur_obj = R_B_ZC_Insur(
    asset_class="R",
    security_type="B_ZC",
    ticker=s_ticker
)
R_B_ZC_Insur_obj.n = 3
#
# st.write(R_B_ZC_Insur_obj)
#
# c_display, c_text, c_input, c_formula = st.columns([3, 2, 1, 1])
# with c_display:
#     st.latex(R_B_ZC_Insur_obj.s_display_formula)
# with c_text:
#     st.latex(R_B_ZC_Insur_obj.s_text_fields)
# with c_input:
#     try:
#
#         n = R_B_ZC_Insur_obj.n
#         s_n = st.text_input(':green[Input: Time Horizon [year]:]', str(n))
#         m = 1 / n
#         nm = n * m
#         per = 1 / m
#
#         st.text('\n')
#         s_m = st.text_input('Output: Coupon Freq [1/year]:', m)
#         s_nm = st.text_input('Output: Total Num Coupons [period]:', str(nm))
#         st.text('\n')
#         s_per = st.text_input('Output: Period Length [year]:', str(per))
#         st.text('\n')
#         st.text('\n')
#
#         # https://eservices.mas.gov.sg/statistics/fdanet/BenchmarkPricesAndYields.aspx
#         # r_R_B_Govt_SG_3y     = 3.16   <==   2y = 3.24, 5y = 2.99
#         r_R_B_Govt_SG_2y = 3.24
#         r_R_B_Govt_SG_5y = 2.99
#         r_R_B_Govt_SG_3y = r_R_B_Govt_SG_2y + 1 / 3 * (r_R_B_Govt_SG_5y - r_R_B_Govt_SG_2y)  # = 3.15
#         r_R_B_Bank_SG_1y_BOC = 3.10  # roll to year 2 and 3, rate not guaranteed
#         r_R_B_Bank_SG_1y_RHB = 3.25  # roll to year 2 and 3, rate not guaranteed
#         r_R_B_SG_Best = r_R_B_Bank_SG_1y_RHB
#
#         # st.markdown("""<style>.st-eb {background-color: black;}</style>""", unsafe_allow_html=True)
#         s_r = st.text_input(':green[Input: Risk-Free Rate [%/year]:]', r_R_B_SG_Best)
#         r = float(s_r) / 100
#
#         r_per = r / m
#         s_r_per = st.text_input('Output: Risk-Free Rate [%/period]:', str(r_per * 100))
#         # :color[green]
#
#         st.text('\n')
#         st.text('\n')
#         s_c = st.text_input(':green[Input: Coupon Rate [%/year]:]', 3.62)
#         c = float(s_c) / 100
#         c_per = c / m
#         s_c_per = st.text_input('Output: Coupon Rate [%/period]:', str(c_per * 100))
#
#         s_N = st.text_input(':green[Input: Notional(t=0) [SGD]:]', "{:,.2f}".format(20000))
#         N = float(s_N.replace(",", ""))
#
#         # CF = c * N
#         # s_CF = st.text_input('Output: CF [SGD/y]:', "{:,.2f}".format(CF))
#         st.text('\n')
#         st.text('\n')
#         CF_per = N * (1 + c_per)
#         s_CF_per = st.text_input(':blue[Output: CF_per [SGD/per]:]', "{:,.2f}".format(CF_per))
#
#         R_B_ZC = 1 / (1 + r_per) * CF_per
#         s_R_B_ZC = st.text_input(':blue[Output: PV of Zero-Coupon [SGD]:]', "{:,.2f}".format(R_B_ZC))
#
#         st.text('\n')
#         st.text('\n')
#         f_comm_FA_pct = R_B_ZC_Insur_obj.f_comm_pct_prem_plus_gst_to_FA / 100
#         f_comm_FA_amt = f_comm_FA_pct * N
#         s_f_comm_FA_amt = st.text_input(':blue[Output: Commission to FA [SGD]:]', "{:,.2f}".format(f_comm_FA_amt))
#
#     except Exception as e:
#         st.write(e)
#         pass
# with c_formula:
#     st.write('\n')
#     st.write('\n')
#     st.write(n)
#     st.write('\n')
#     st.write('\n')
#     st.write('\n')
#     st.write(m)
#     st.write('\n')
#     st.write('\n')
#     st.write('\n')
#     st.write(nm)
#     st.write('\n')
#     st.write('\n')
#     st.write('\n')
#     st.write('\n')
#     st.write(per)
#     st.write('\n')
#     st.write('\n')
#     st.write('\n')
#     st.write('\n')
#     st.write('\n')
#     st.write(r)
#     st.write('\n')
#     st.write('\n')
#     st.write('\n')
#     st.write(r_per)
#     st.write('\n')
#     st.write('\n')
#     st.write('\n')
#     st.write('\n')
#     st.write(c)
#     st.write('\n')
#     st.write('\n')
#     st.write('\n')
#     st.write(c_per)
#     st.write('\n')
#     st.write('\n')
#     st.write('\n')
#     st.write(N)
#     # st.write(CF)
#     st.write('\n')
#     st.write('\n')
#     st.write('\n')
#     st.write('\n')
#     st.write(CF_per)
#     st.write('\n')
#     st.write('\n')
#     st.write('\n')
#     st.write(R_B_ZC)
#     st.write('\n')
#     st.write('\n')
#     st.write('\n')
#     st.write('\n')
#     st.write('\n')
#     st.write(f_comm_FA_amt)
#
# with c_display:
#     c_field, c_data = st.columns([1, 1])
#     with c_field:
#         st.write("Date of Launch")
#     with c_data:
#         st.write(R_B_ZC_Insur_obj.dt_launch)
#
#     with c_field:
#         st.write("Policy Term [years]")
#     with c_data:
#         st.write(R_B_ZC_Insur_obj.i_policy_term_in_years)
#     with c_field:
#         st.write("Time Horizon Min [years]")
#     with c_data:
#         st.write(R_B_ZC_Insur_obj.i_time_horizon_years_min)
#     with c_field:
#         st.write("Time Horizon Max [years]")
#     with c_data:
#         st.write(R_B_ZC_Insur_obj.i_time_horizon_years_max)
#
#     with c_field:
#         st.write("Return Guaranteed")
#     with c_data:
#         st.write(R_B_ZC_Insur_obj.s_return_guaranteed)
#
#     with c_field:
#         st.write("Issuer")
#     with c_data:
#         st.write(R_B_ZC_Insur_obj.s_issuer)
#     with c_field:
#         st.write("Product or Plan")
#     with c_data:
#         st.write(R_B_ZC_Insur_obj.s_product_plan)
#     with c_field:
#         st.write("Premium Type")
#     with c_data:
#         st.write(R_B_ZC_Insur_obj.s_premium_type)
#     with c_field:
#         st.write("Premium Min [" + R_B_ZC_Insur_obj.s_currency + "]")
#     with c_data:
#         st.write("{:,.2f}".format(R_B_ZC_Insur_obj.f_premium_SGD_min))
#     with c_field:
#         st.write("Premium Incr [" + R_B_ZC_Insur_obj.s_currency + "]")
#     with c_data:
#         st.write("{:,.2f}".format(R_B_ZC_Insur_obj.f_premium_SGD_step))
#     with c_field:
#         st.write("Premium Max [" + R_B_ZC_Insur_obj.s_currency + "]")
#     with c_data:
#         st.write("{:,.2f}".format(R_B_ZC_Insur_obj.f_premium_SGD_max))
#     with c_field:
#         st.write("Participation")
#     with c_data:
#         st.write(R_B_ZC_Insur_obj.s_par)
#     with c_field:
#         st.write("s_eligibility_age_min")
#     with c_data:
#         st.write(R_B_ZC_Insur_obj.s_eligibility_age_min)
#     with c_field:
#         st.write("s_eligibility_age_max")
#     with c_data:
#         st.write(R_B_ZC_Insur_obj.s_eligibility_age_max)
#     with c_field:
#         st.write("s_eligibility_residency_SG_Citizen")
#     with c_data:
#         st.write(R_B_ZC_Insur_obj.s_eligibility_residency_SG_Citizen)
#     with c_field:
#         st.write("s_eligibility_residency_SG_PR")
#     with c_data:
#         st.write(R_B_ZC_Insur_obj.s_eligibility_residency_SG_PR)
#     with c_field:
#         st.write("s_eligibility_residency_SG_EP")
#     with c_data:
#         st.write(R_B_ZC_Insur_obj.s_eligibility_residency_SG_EP)
#     with c_field:
#         st.write("s_issuer_contact_number")
#     with c_data:
#         st.write(R_B_ZC_Insur_obj.s_issuer_contact_number)
#     with c_field:
#         st.write("s_issuer_contact_email")
#     with c_data:
#         st.write(R_B_ZC_Insur_obj.s_issuer_contact_email)
#
#     with c_field:
#         st.write("Product Description")
#     with c_data:
#         st.write(R_B_ZC_Insur_obj.s_product_desc)

# s_product_benefit_maturity = (
#     "If the insured survives at the end of the policy term and this policy has not ended,  \n"
#     "we will pay the guaranteed maturity benefit at the end of the policy term.  \n"
#     "This policy will end when we make this payment.")
# s_product_benefit_death = "In the event of death of the life assured, the single premium is returned with interest, using the guaranteed simple interest rate."
# s_product_benefit_death_accidental = ("In the event of accidental death in the first year of the policy,  \n"
#                                       "subject to the policyholder being under age 70,  \n"
#                                       "an additional 10% of the single premium will be payable.")
# s_product_benefit_TPD = (
#     "Upon diagnosis of TPD before age 65, the single premium is returned with interest, using the guaranteed simple interest rate.")
# s_product_benefit_exclusions = ("Death due to suicide within one year from the date of issue of Policy."


#============================================================================================================================================
# main
#============================================================================================================================================
if __name__ == "__main__":
    pass

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

st.set_page_config(page_title="R_B_ZC_Insur_01", page_icon="📹")
st.markdown("# R_B_ZC_Insur")

# st.sidebar.header("R_B_ZC_Insur")
st.write(
    """ LIC
    """
)

# Create the Object using Standard Parameters
s_ticker = r"R_B_ZC_Insur___NKSgd=020-200_Prem1_T=3y_r=3.62\%_Endow_Par0_LIC_WealthPlus3.1"
R_B_ZC_Insur_obj = _Class_R_B_ZC_Insur._Class_R_B_ZC_Insur(
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


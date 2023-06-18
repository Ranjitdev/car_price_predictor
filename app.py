from src.utils import StreamliDataProvider
from src.pipeline.prediction_pipeline import PredictPipeline
import streamlit as st


class WebForm(StreamliDataProvider):
    def __init__(self):
        StreamliDataProvider.__init__(self)

    def streamlit_form(self):
        with st.form('form'):
            car_name = st.selectbox(
                ':blue[Select car model]', self.car_names
            )
            location = st.selectbox(
                ':blue[Select car owner location]', self.locations
            )
            km_driven = st.number_input(':blue[Car driven in km]')
            fuel = st.selectbox(
                ':blue[Select car fuel type]', self.fuels
            )
            owner = st.slider(
                ':blue[First hand, Second hand or more]', 1, 5
            )
            year = st.slider(
                ':calendar: :blue[Select car buying year]', 2000, 2022
            )

            submitted = st.form_submit_button('Get car price')
            if submitted:
                if km_driven > 0:
                    st.write('Model ', car_name)
                    st.write('Car from ', location)
                    st.write('Total driven ', km_driven, 'km')
                    st.write('Fuel type ', fuel)
                    st.write('Previous owners', owner)
                    st.write('Car bought on ', year)
                    df = PredictPipeline(
                        car_name, location, km_driven, fuel, owner, year
                    ).data_to_df()
                    prediction = PredictPipeline().predict_result(df)
                    st.subheader(f'This car can cost you approximately :green[_{prediction}_]')


WebForm().streamlit_form()
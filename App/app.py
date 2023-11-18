from shiny import App, ui, reactive, render
from src.modelFunctions import get_predictions
import pandas as pd
import shinyswatch
from pathlib import Path

css_path = Path(__file__).parent/ "www" /"my-style.css"

# Part 1: ui ----
app_ui = ui.page_fluid(
    ui.include_css(css_path),
    #shinyswatch.theme.lux(),
    ui.div(
        ui.panel_title("Somm Wine Reccomendation Engine", window_title="Somm"),
        class_="centered-text title-margin"
    ),
    ui.div(
        ui.input_text("user_query",
                 "Describe your wine",
                  value='',
                  width='400px', 
                  placeholder="A fruity wine with notes of plum", 
                  autocomplete='off', 
                  spellcheck='true'),
        ui.input_action_button(id='submit', label='Submit', class_= "submit-button"),
        class_="centered-container"
    ),
    ui.output_table("recommendation"),
)
# Part 2: server ----
# coordinates inputs and outputs
def server(input, output, session):
    output_table = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.submit)
    def on_submit():
        query = input.user_query()
        if query:  # Ensure there is a query to process
            # Call the get_predictions function
            predictions_df = get_predictions(query) #pd.DataFrame(predictions_array, columns=["Description"])
            output_table.set(predictions_df)
        else:
            # Return an empty DataFrame or None if there is no query
            output_table.set(pd.DataFrame())
    
    @output
    @render.table
    def recommendation():
        return output_table.get()

# Combine into a shiny app.
# Note that the variable must be "app".
app = App(app_ui, server)

"""
-navbar with image/pictures
    -a logo: something related to a somollier
    - static image of wine holding wine glass minimalist
    -make it animated?
    Palak

-a realistic banner image
-the search bar infront of the banner
-a tagline the search
-make sure the resutls display underneath the banner

-beautify how we display the results
-tagline what makes us different?: the power of ML + 130k critics
-another tab: present the most popular searched wines this week
    -get their feedback, rate this output thumbs up thumbs down
    - wine glass standing up, wine glass spilled 
-pre-stored recommendations for red wines, and white wines, sparkling
    -filters for the output
-Visualizaton
    -wordcloud

"""
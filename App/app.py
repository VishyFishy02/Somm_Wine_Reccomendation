from shiny import App, ui, reactive, render
from src.rag_model_functions import get_predictions, get_wine_styles
import pandas as pd
from pathlib import Path
import mistune

css_path = Path(__file__).parent/ "www" /"my-style.css"
www_dir =  Path(__file__).parent/ "www"
# Part 1: ui ----
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.link(
                rel="stylesheet", 
                href="https://fonts.googleapis.com/css?family=Lato:400,700"
            )
    ),
    ui.include_css(css_path),
    ui.div(
        ui.img(src="logo.png", class_="logo-image", height="20", width="auto"),
        ui.page_navbar(
            title="Som·me·lier: a wine steward.",
            bg="#6f1414"
        ),
        class_= "navbar"
    ),
    ui.div( # outer banner div
        ui.div(
            ui.panel_title("Somm Wine Recommendation Engine", window_title="Somm"),
            class_="title-text title-margin"
        ),
        ui.div( # inner searchbar div
            ui.input_text("user_query",
                    "The power of 130k wine experts and AI",
                    value='',
                    width='400px', 
                    placeholder="A fruity wine with notes of plum", 
                    autocomplete='off', 
                    spellcheck='true'),
            ui.input_action_button(id='submit', label='Enter', class_= "submit-button"),
            class_="search-container"
        ),
        class_="centered-container",
        style="background-image: url('banner3.png'); background-size: cover;"
    ),
    ui.navset_tab(
        ui.nav("Wines", ui.output_table("recommendation")),
        ui.nav("Styles", ui.output_table("styles", align="center", class_="styles-table")),
        id="data_tabs"
    ),
)
# Part 2: server ----
# coordinates inputs and outputs
def server(input, output, session):
    output_table = reactive.Value(None)
    output_styles = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.submit)
    def on_submit():
        print("submit button pressed")
        query = input.user_query()
        if query:  # Ensure there is a query to process
            
            # Call the get_predictions function
            predictions_df = get_predictions(query) #pd.DataFrame(predictions_array, columns=["Description"])
            styles_df = get_wine_styles(query)
            output_table.set(predictions_df)
            output_styles.set(styles_df)
            print("result recieved!")

        else:
            # Return an empty DataFrame or None if there is no query
            output_table.set(pd.DataFrame())
            output_styles.set(pd.DataFrame())
    
    @output
    @render.table
    def recommendation():
        return output_table.get()

    @output
    @render.table
    def styles():
        return output_styles.get()

# Combine into a shiny app.
# Note that the variable must be "app".
app = App(app_ui, server, static_assets=www_dir)

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
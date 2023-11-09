from shiny import App, ui, reactive, render
from src.modelFunctions import get_predictions
import pandas as pd
# Part 1: ui ----
app_ui = ui.page_fluid(
    ui.panel_title("Somm Wine Reccomendation Engine", window_title="Somm"),
    ui.input_text("user_query",
                 "What are you looking for?:",
                  value='',
                  width='400px', 
                  placeholder="ex. A fruity wine that is not bitter", 
                  autocomplete='off', 
                  spellcheck='true'),
    ui.input_action_button(id='submit', label='Submit'),
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
            predictions_array = get_predictions(query)
            # Convert the numpy array to a pandas DataFrame
            predictions_df = pd.DataFrame(predictions_array, columns=["Description"])
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
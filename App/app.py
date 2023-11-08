from shiny import App, ui

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
    ui.output_table("reccomendation"),
)
# Part 2: server ----
def server(input, output, session):
    ...

# Combine into a shiny app.
# Note that the variable must be "app".
app = App(app_ui, server)
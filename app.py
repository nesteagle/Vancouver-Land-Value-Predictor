import gradio as gr
from visualization import plot_neighbourhoods

with gr.Blocks() as demo:
    with gr.Row():
        input_col = gr.Column(scale=1)
        output_col = gr.Column(scale=3)
    
    with input_col:
        instructions = gr.Markdown("Adjust the z-score to visualize results at different amounts of filtering.")
        max_sd_input = gr.Number(label="Max z-score (# SD away from mean)", value=1.25, step=0.25)
        submit_btn = gr.Button("Plot")
    
    with output_col:
        disclaimer = gr.Markdown("Data is sourced from Vancouver OpenData/BCAssessment, and should serve as a rough estimate. No semantic definitions for the neighbourhood codes were provided. Estimations of neighborhood areas are based on this data.", visible=True)
        plot_out = gr.Plot()

    submit_btn.click(fn=plot_neighbourhoods, inputs=max_sd_input, outputs=plot_out)
    demo.load(fn=plot_neighbourhoods, inputs=max_sd_input, outputs=plot_out)

demo.launch()

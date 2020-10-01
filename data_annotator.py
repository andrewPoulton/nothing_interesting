import streamlit as st
# from dataset import SemEvalDataset
import pandas as pd
import sys
from streamlit.components.v1 import components

user = sys.argv[1]
def spanify_word(idx, word):
    return f"<span class='word' id='word_{idx}'>{word}</span>"
def spanify(text):
    return " ".join([spanify_word(i, word) for i, word in enumerate(text.split())])

@st.cache(allow_output_mutation=True)
def init():
    df = pd.read_csv('data/flat_semeval5way_test.csv')
    df = df[(df.origin == 'unseen_answers')&(df.source == 'scientsbank')]
    df['spans'] = df['student_answers'].apply(spanify)
    return df

data = init()


_css = r"""
<style>
    .word:hover{
        background-color: green;
        color: white;
        cursor: pointer;
    }
</style>
"""
_component_func = components.declare_component(
        # We give the component a simple, descriptive name ("my_component"
        # does not fit this bill, so please choose something better for your
        # own component :)
        "my_component",
        # Pass `url` here to tell Streamlit that the component will be served
        # by the local dev server that you run via `npm run start`.
        # (This is useful while your component is in development.)
        url="http://localhost:3001",
    )

def my_component(name, key=None):
    """Create a new instance of "my_component".

    Parameters
    ----------
    name: str
        The name of the thing we're saying hello to. The component will display
        the text "Hello, {name}!"
    key: str or None
        An optional key that uniquely identifies this component. If this is
        None, and the component's arguments are changed, the component will
        be re-mounted in the Streamlit frontend and lose its current state.

    Returns
    -------
    int
        The number of times the component's "Click Me" button has been clicked.
        (This is the value passed to `Streamlit.setComponentValue` on the
        frontend.)

    """
    # Call through to our private component function. Arguments we pass here
    # will be sent to the frontend, where they'll be available in an "args"
    # dictionary.
    #
    # "default" is a special argument that specifies the initial return
    # value of the component before the user has interacted with it.
    component_value = _component_func(name=name, key=key, default=[])

    # We could modify the value returned from the component if we wanted.
    # There's no need to do this in our simple example - but it's an option.
    return component_value

data_idx = st.empty()
css = st.empty()
st.header("**Question:**")
question = st.empty()
st.header("**Reference answer:**")
reference_answers = st.empty()
st.header("**Student answer:**")

next_idx = data_idx.number_input(
    "Select example", 
    min_value=0, 
    max_value=len(data)-1,
    value=0,
    step=1)
row = data.iloc[next_idx]
css.markdown(_css, unsafe_allow_html=True)
question.markdown(row.question_text)
reference_answers.markdown(row.reference_answers)

annotations = my_component(row.spans)
print(annotations)
# increment_idx = st.button("Next question/student response")

if len(annotations) >0:
    with open(f"annotations/scientsbank_unseen_answers_{next_idx}.annotation", "a") as file:
        file.write(",".join(annotations)+"\n")
    



# print(st.caching._mem_caches._function_caches)
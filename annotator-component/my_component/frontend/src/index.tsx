import { Streamlit, RenderData } from "streamlit-component-lib"

// Add text and a button to the DOM. (You could also add these directly
// to index.html.)
const span = document.body.appendChild(document.createElement("span"))
const textNode = span.appendChild(document.createElement("p"))
const button = span.appendChild(document.createElement("button"))
button.textContent = "Save annotation"
let clicked_words: string[] = []
// Add a click handler to our button. It will send data back to Streamlit.
// let numClicks = 0
button.onclick = function(): void {
  // Increment numClicks, and pass the new value back to
  // Streamlit via `Streamlit.setComponentValue`.
  console.log('clicked_words:', clicked_words)
  if (clicked_words.length > 0){
    Streamlit.setComponentValue(clicked_words)
  }
  clicked_words = []
  Array.from(
    document.getElementsByClassName("label")
    ).forEach(elt => elt.remove())
  Array.from(
    document.getElementsByClassName("clicked")
  ).forEach(elt => elt.className = 'word')
}

function correct_annotations(elt_id: string|null): void {
  if (elt_id===null){return}
  let label_index = clicked_words.indexOf(elt_id)
  clicked_words = clicked_words.slice(0, label_index).concat(clicked_words.slice(label_index+1))
  let labels: Element[] = Array.from(document.getElementsByClassName("label"))
  for (var i = 0; i <labels.length; i++){
    let current_label: number = Number(labels[i].textContent)
    if (current_label != null) {
      current_label -= current_label > label_index?1:0
      labels[i].textContent = current_label + ""
    }
  }
}


function word_clicked(this: HTMLSpanElement): void {
  console.log("clicked!", this)
  if (this.className === 'word'){
    clicked_words.push(this.id)
    this.className = "clicked"
    this.innerHTML += ` <span class='label'>${clicked_words.length}</span>`
  } else if (this.className === 'clicked'){
    this.innerHTML = this.innerHTML.split(" ")[0]
    this.className = 'word'
    correct_annotations(this.id)
  }
  console.log(clicked_words)
  Streamlit.setFrameHeight()
}

/**
 * The component's render function. This will be called immediately after
 * the component is initially loaded, and then again every time the
 * component gets new data from Python.
 */
function onRender(event: Event): void {
  // Get the RenderData from the event
  const data = (event as CustomEvent<RenderData>).detail

  // Disable our button if necessary.
  button.disabled = data.disabled

  // empty clicked_words
  clicked_words = []
  // RenderData.args is the JSON dictionary of arguments sent from the
  // Python script.
  let name = data.args["name"]

  // Show "Hello, name!" with a non-breaking space afterwards.
  textNode.innerHTML = `${name}!` + String.fromCharCode(160)

  let elts = document.getElementsByClassName('word')
  Array.from(elts).forEach(elt => elt.addEventListener('click', word_clicked))

  // We tell Streamlit to update our frameHeight after each render event, in
  // case it has changed. (This isn't strictly necessary for the example
  // because our height stays fixed, but this is a low-cost function, so
  // there's no harm in doing it redundantly.)
  Streamlit.setFrameHeight()
}

// Attach our `onRender` handler to Streamlit's render event.
Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender)

// Tell Streamlit we're ready to start receiving data. We won't get our
// first RENDER_EVENT until we call this function.
Streamlit.setComponentReady()

// Finally, tell Streamlit to update our initial height. We omit the
// `height` parameter here to have it default to our scrollHeight.
Streamlit.setFrameHeight()

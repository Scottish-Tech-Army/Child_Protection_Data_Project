/* eslint-disable */
<template>
  <div class="appContainer">
    <div class="inputContainer">
      <div class="chooseModelContainer">
        <h1>Select the model:</h1>
        <model_menu
          @modelChanged="modelSelected = $event"
          :disabled="isLoading ? true : false"
        />
      </div>
      <div class="textsContainer">
        <textarea
          class="input"
          v-model="textToSummarize"
          rows="20"
          cols="30"
          placeholder="Text to summarize..."
          :disabled="isLoading ? true : false"
        >
        </textarea>
        <div class="overlay">
          <textarea
            v-if="!isLoading"
            v-model="output"
            class="input"
            rows="20"
            cols="30"
            :disabled="output ? false : true"
          >
          </textarea>
          <div v-else class="breeding-rhombus-spinner">
            <div class="rhombus child-1"></div>
            <div class="rhombus child-2"></div>
            <div class="rhombus child-3"></div>
            <div class="rhombus child-4"></div>
            <div class="rhombus child-5"></div>
            <div class="rhombus child-6"></div>
            <div class="rhombus child-7"></div>
            <div class="rhombus child-8"></div>
            <div class="rhombus big"></div>
          </div>
        </div>
      </div>
      <button
        @click="executeSummarization"
        class="buttonSummarize"
        type="button"
      >
        Summarize!
      </button>
    </div>
    <div class="outputContainer">
      <iframe class="i-frame" :srcdoc="html_ner"></iframe>
      <iframe class="i-frame" :srcdoc="html_sentiment"></iframe>
    </div>
  </div>
</template>

<script>
// @ is an alias to /src
import model_menu from "@/components/Model_menu.vue";
export default {
  name: "HelloWorld",
  data() {
    return {
      //output,
      output: "",
      textToSummarize: "",
      modelSelected: "",
      isLoading: false,
      html_ner: "",
      html_sentiment: "",
    };
  },
  components: {
    model_menu,
  },
  methods: {
    async getJSON() {
      this.isLoading = true;
      await fetch("http://localhost:5006/set", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: this.textToSummarize,
          model: this.modelSelected,
        }),
      })
        .then((resp) => resp.json())
        .then((data) => {
          this.output = data.text;
          this.html_ner = data.html_ner;
          this.html_sentiment = data.html_sentiment;
        })
        .catch((error) => console.error(error))
        .finally(() => (this.isLoading = false));
    },
    executeSummarization() {
      this.output = "";
      //console.log(this.modelSelected);
      if (this.modelSelected !== "") {
        this.getJSON();
      } else {
        alert("No model selected.");
      }
    },
  },
  //computed() {
  //  this.modelSelected = model_menu.data().itemSelected;
  //},
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
@import "@/styles/spinner.css";
h3 {
  margin: 40px 0 0;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
.input {
  resize: none;
}
.appContainer {
  display: flex;
  justify-content: center;
  width: 100%;
  gap: 25px;
}
.inputContainer {
  width: 600px;
  margin: 30px auto;
  min-height: 300px;
  border: 2px solid steelblue;
  border-radius: 5px;
  padding: 30px;
}
.chooseModelContainer {
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 20px;
  gap: 30px;
}
.textsContainer {
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 20px;
  gap: 50px;
}
.buttonSummarize {
  width: 400px;
  height: 50px;
}
.overlay {
  width: 250px;
}
.outputContainer {
  display: flex;
  justify-content: space-evenly;
  flex-direction: column;
  width: 100%;
  gap: 28px;
}
.i-frame {
  display: flex;
  width: 95%;
  height: 40%;
  border: 2px solid steelblue;
  border-radius: 5px;
}
</style>

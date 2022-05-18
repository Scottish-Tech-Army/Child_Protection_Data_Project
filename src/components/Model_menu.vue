<template>
  <section class="dropDownMenuWrapper">
    <button class="dropDownMenuButton" ref="menu" @click="openClose">
      {{ itemSelected }}
    </button>
    <div class="iconWrapper">
      <div class="bar1" :class="{ 'bar1--open': isOpen }" />
      <div class="bar2" :class="{ 'bar2--open': isOpen }" />
      <div class="bar3" :class="{ 'bar3--open': isOpen }" />
    </div>
    <section class="dropdownMenu" v-if="isOpen">
      <section
        class="option"
        v-for="m in models"
        :key="m.name"
        @click.prevent="itemClick(m)"
      >
        <a>{{ m.name }}</a>
      </section>
    </section>
  </section>
</template>

<script>
export default {
  name: "model_menu",
  data() {
    return {
      itemSelected: "Model",
      isOpen: false,
      models: [
        { name: "Pegasus", link: "false" },
        { name: "DistilBart1", link: "true" },
        { name: "DistilBart2", link: "true" },
        { name: "Bart1", link: "true" },
        { name: "Bart2", link: "true" },
      ],
    };
  },
  methods: {
    openClose() {
      var _this = this;
      const closeListerner = (e) => {
        if (_this.catchOutsideClick(e, _this.$refs.menu))
          window.removeEventListener("click", closeListerner),
            (_this.isOpen = false);
      };
      window.addEventListener("click", closeListerner);
      this.isOpen = !this.isOpen;
    },
    catchOutsideClick(event, dropdown) {
      // When user clicks menu — do nothing
      if (dropdown == event.target) return false;
      // When user clicks outside of the menu — close the menu
      if (this.isOpen && dropdown != event.target) return true;
    },
    itemClick(item) {
      this.itemSelected = item.name;
      this.$emit("modelChanged", this.itemSelected);
    },
  },
};
</script>
<style lang="scss" scoped>
@import "@/styles/model_menu.scss";
</style>

require('./bootstrap');


import BootstrapVue from "bootstrap-vue"
import 'bootstrap/dist/css/bootstrap.css'
import 'bootstrap-vue/dist/bootstrap-vue.css'
 
require('./bootstrap');
 
const Vue = require('vue');

Vue.use(BootstrapVue)

window.Vue = Vue

import NumberComponent from './NumberComponent.vue'

const app = new Vue({
    components: {
        NumberComponent,
    },
    el: '#app',
})

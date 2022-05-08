<template>
    <b-card title="数字判定">
        <!-- <h1></h1> -->
        <b-card-text>
            <b-row>
                <b-col>
                    <canvas
                        style="border:1px solid black; width: 280px; height: 280px;" 
                        ref="canvas"
                        width="28" height="28"
                        @mousedown.left="onMouseDown"
                        @mouseup.left="onMouseUp"
                        @mousemove="paint"
                    ></canvas>
                </b-col>
            </b-row>
            <b-row class="result-box">
                <b-col
                    
                >
                    <p
                        v-if="result"
                    >
                        数字は<span class="num-text">{{ result.num }}</span>です！
                    </p>
                </b-col>
            </b-row>
            <b-row>
                <b-col>
                    <b-button
                        variant="primary" 
                        @click="click"
                    >
                        どの数字？
                    </b-button>
                </b-col>
                <b-col>
                    <b-button variant="danger" 
                        @click="clear"
                    >
                        消す
                    </b-button>
                </b-col>
            </b-row>
        </b-card-text>
    </b-card>
</template>

<script>
export default {
    moutend() {
        this.canvas = this.$refs.canvas;


    },
    name: 'number-component',

    data() {
        return {
            canvas: null,
            context: null,
            isPainting: false,
            mouseX: null,
            mouseY: null,
            result: null,
        }
    },

    methods: {
        click() {
            const context = this.$refs.canvas.getContext("2d");
            const dat = context.getImageData(0, 0, 28, 28)

            const pix = []
            for (let i = 0; i < (28 * 28 * 4); i+= 4) {
                pix.push(dat.data[i + 3] / 255.0)
            }
// console.log(pix)
            const axios = window.axios.create({
                responseType: 'json',
            });
            axios.post('/num', {
                dat: pix,
            }).then((res) => {
                this.result =  res.data
            })
        },
        clear() {
            const context = this.$refs.canvas.getContext("2d");
            context.clearRect(0, 0, 28, 28)
            this.result = null

        },
        onMouseDown(e) {
            this.isPainting = true
            // const canvas = this.$refs.canvas
            // const rect = canvas.getBoundingClientRect();

            this.mouseX = e.offsetX // - rect.left;
            this.mouseY = e.offsetY // - rect.top;
        },
        onMouseUp(e) {
            this.isPainting = false
            this.mouseX = null
            this.mosueY = null
        },
        paint(e) {
            if (!this.isPainting) {
                return
            }
            

            // const canvas = this.$refs.canvas
            // const rect = canvas.getBoundingClientRect();
            const mouseX = e.offsetX //- rect.left
            const mouseY = e.offsetY //- rect.top

            this.context = this.$refs.canvas.getContext("2d");
            this.context.lineWidth = 1
            this.context.lineCap = 'round'
            this.context.beginPath()
            this.context.moveTo(this.mouseX / 10, this.mouseY / 10)
            this.context.lineTo(mouseX / 10, mouseY / 10)
            this.context.stroke()
            
            this.mouseX = mouseX
            this.mouseY = mouseY
        },
    },    
}
</script>

<style lang="scss" scoped>
.result-box {
    height: 80px;
}
.num-text {
    font-size: 160%;
}
</style>
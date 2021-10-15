let brain;
let color = { r: 0, b: 0, g: 0 };
let brainGuess;
let isTrained = false;
let gen = 0;
function setup() {
	createCanvas(800, 800);

	pickColor();

	//3 inputs (r,g,b)
	//3 hidden like the inputs
	//2 output (0:probability of black, 1:probability of white)
	brain = new NeuralNetwork(3, 3, 2);
}

function draw() {
	// frameRate(10);
	background(0);

	noStroke();
	textAlign(CENTER, CENTER);

	if (!isTrained) {
		train();
	} else {
		process();
	}
}

function mousePressed() {
	if (isTrained) {
		pickColor();
		brainGuess = colorGuesser(color);
		noLoop();
		redraw();
	}
}

function keyPressed() {
	isTrained = checkIsTrained();
	console.log(isTrained);
}

const process = () => {
	fill(color.r, color.g, color.b);
	rect(0, 200, width, height - 400);
	fill(255);
	textSize(20);
	text(`Trained in ${gen} generations`, 400, 10);
	textSize(60);
	fill(0);
	text('black', 250, 390);
	fill(255);
	text('white', 550, 390);
	textSize(40);
	text('Press the mouse button to take a guess', 400, 100);

	if (brainGuess) {
		textSize(30);
		text(`I think the right text color for this background is ${brainGuess}`, 400, 700);
		if (brainGuess === 'black') {
			fill(0);
			ellipse(250, 490, 60);
		} else {
			fill(255);
			ellipse(550, 490, 60);
		}
	}
};

const train = () => {
	background(color.r, color.g, color.b);
	textSize(60);
	text('training the brain...', 400, 390);
	textSize(20);
	text(gen, 400, 10);

	for (let i = 0; i < 100; i++) {
		gen++;
		pickColor();
		brain.train(getInputArray(color), getTargetArray(colorPredictor(color)));
	}
	isTrained = checkIsTrained();
	if (isTrained) {
	}
};

// no ML
const colorPredictor = ({ r, g, b }) => (r + g + b > 300 ? 'black' : 'white');

const colorGuesser = (c) => {
	let outputs = brain.feedforward(getInputArray(c));
	return outputs[0] > outputs[1] ? 'black' : 'white';
};

const checkIsTrained = () => {
	for (let i = 0; i < 50; i++) {
		let c = getRandomColor();
		let guess = colorGuesser(c);
		let target = colorPredictor(c);
		if (guess !== target) {
			return false;
		}
	}
	return true;
};

//normalize inputs to range 0-1
const getInputArray = ({ r, g, b }) => [r / 255, g / 255, b / 255];

const getTargetArray = (target) => (target === 'black' ? [1, 0] : [0, 1]);

const getRandomColor = () => ({ r: random(255), g: random(255), b: random(255) });

const pickColor = () => {
	color = getRandomColor();
};

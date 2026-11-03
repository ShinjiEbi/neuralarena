// Worker d'entraînement : la simulation tourne hors du fil de la page,
// donc sans à-coups pour l'interface et sans le bridage des onglets en arrière-plan.
importScripts('sim-core.js');
const store = new Store();
const driver = new TrainingDriver(msg => postMessage(msg), store);
onmessage = e => driver.handle(e.data);
postMessage({type:'ready'});

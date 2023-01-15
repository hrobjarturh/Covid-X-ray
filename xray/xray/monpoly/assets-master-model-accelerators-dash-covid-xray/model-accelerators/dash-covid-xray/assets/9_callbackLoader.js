//alert('If you see this alert, then your custom JavaScript script has run!')
// Select the node that will be observed for mutations
const targetNode = document.getElementById('react-entry-point');

// Options for the observer (which mutations to observe)
const config = { attributes: false, childList: true, subtree: true };

// Callback function to execute when mutations are observed
const callback = function(mutationList, observer) {
    const targetNode = document.querySelector('._dash-loading-callback');
    console.log(targetNode)
    if (targetNode) {
        console.log(targetNode)

        targetNode.setAttribute('data-dash-initial-loading', 'true')
        // Later, you can stop observing
        observer.disconnect();
    }
};

// Create an observer instance linked to the callback function
const observer = new MutationObserver(callback);

// Start observing the target node for configured mutations
observer.observe(targetNode, config);



## Gaussian Probability Divergences

To avoid excess detail in the main readme, we detail the specifics of our additional patch-wise complexity measures.

We assume that our image has been separated into patches, and we have fit distributional moments to each one, resulting in an empirical mean $`\widehat{\mu}(p)\in\mathbb{R}^{3}`$ and covariance $`\widehat{\Sigma}(p)\in\mathbb{R}^{3\times 3}`$ for each patch $`p`$.

Following the natural Maximum Entropy assumption [1], we consider each patch to be multivariate Gaussian $`\mathcal{G} = \mathcal{N}(\widehat{\mu}(p),\widehat{\Sigma}(p))`$, and assume that a complex image is likely to contain patches are as "different" from each other as possible.
We therefore measure the expected pairwise distance between patches as a simple complexity measure.
Since trivial vector space metrics (e.g., the $`L_2`$ distance) are well-known to be poor image metrics in the context of biological visual perception [2], we utilize distributional divergences instead, which are more robust to perceptually irrelevant perturbations, such as small translations.

Thus, we consider a family of complexity measures on an image $`I`$ defined by
```math
\mathcal{C}_{\mathfrak{D}}(I) = \frac{1}{|P|^2} \sum_{p_i,p_j\in P} \mathfrak{D}\left[ \mathcal{G}_i \mid\mid \mathcal{G}_j \right],
```
where $`P`$ is the set of patches, $`\mathcal{G}_i`$ is the Gaussian corresponding to patch $`i`$, and $`\mathfrak{D}`$ is a particular parameterizing choice of information-theoretic divergence, the options for which are detailed below. Note that these divergences (and hence the resulting metrics) are simple to compute and computationally tractable, due to the Gaussian assumption.

### Jeffrey's Divergence


### Wasserstein-2 Metric 


### Bhattacharyya distance

For notational simplicity, define the Bhattacharyya coefficient
```math
\mathcal{B}[\mathcal{P},\mathcal{Q}] = \int \sqrt{p(x) q(x)} \, dx,
```
where $`\mathcal{P},\mathcal{Q}`$ are probability distributions over $`x`$, and $`p,q`$ are their respective densities.

Then the Bhattacharyya distance is simply given by
```math
\mathfrak{D}_B\left[ \mathcal{G}_i \mid\mid \mathcal{G}_j \right] = -\log\mathcal{B}[\mathcal{G}_i, \mathcal{G}_j].
```

### Squared Hellinger distance

Utilizing the notation just above, the squared Hellinger distance is given by 
```math
\mathfrak{D}_B\left[ \mathcal{G}_i \mid\mid \mathcal{G}_j \right] = \sqrt{ 1 - \mathcal{B}[\mathcal{G}_i, \mathcal{G}_j] }.
```

### Forstner-Moonen Abou-Moustafa-Torres-Ferries (FM-ATF) density metric

Folllowing [3,4], we utilize the following metric



## References

[1] Jaynes, Edwin T. "Information theory and statistical mechanics." Physical review 106.4 (1957): 620.

[2] Zhang, Richard, et al. "The unreasonable effectiveness of deep features as a perceptual metric." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

[3] Abou–Moustafa, Karim T., Fernando De La Torre, and Frank P. Ferrie. "Designing a metric for the difference between Gaussian densities." Brain, Body and Machine. Springer, Berlin, Heidelberg, 2010. 57-70.

[4] Förstner, Wolfgang, and Boudewijn Moonen. "A metric for covariance matrices." Geodesy-the Challenge of the 3rd Millennium. Springer, Berlin, Heidelberg, 2003. 299-309.


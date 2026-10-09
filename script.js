// Simple intersection observer to add 'in-view' class to feature cards
const observerOptions = {
  root: null,
  rootMargin: '0px',
  threshold: 0.3
}

const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('in-view');
    }
  });
}, observerOptions);

document.querySelectorAll('.feature-card').forEach(card => {
  observer.observe(card);
});

// Parallax effect for hero image (optional)
window.addEventListener('scroll', () => {
  const scrolled = window.pageYOffset;
  const image = document.querySelector('.car-illustration');
  if (image) {
    image.style.transform = `translateY(${scrolled * 0.3}px)`;
  }
});

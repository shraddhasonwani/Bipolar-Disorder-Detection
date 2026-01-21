// Main website JavaScript
document.addEventListener('DOMContentLoaded', function () {
    initializeWebsite();
    setupScrollEffects();
    setupMobileMenu();
    setupAnimations();
});

/* ================= INITIALIZE ================= */
function initializeWebsite() {
    // Smooth scrolling ONLY for internal section links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const targetId = this.getAttribute('href');

            if (targetId.length > 1) {
                const target = document.querySelector(targetId);
                if (target) {
                    e.preventDefault();
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });

    const contactForm = document.querySelector('.contact-form form');
    if (contactForm) {
        contactForm.addEventListener('submit', handleContactForm);
    }

    setupStatsAnimation();
}

/* ================= NAVBAR SCROLL ================= */
function setupScrollEffects() {
    const navbar = document.querySelector('.navbar');
    if (!navbar) return;

    window.addEventListener('scroll', () => {
        navbar.style.background =
            window.scrollY > 100
                ? 'rgba(255, 255, 255, 0.98)'
                : 'rgba(255, 255, 255, 0.95)';
    });

    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, { threshold: 0.1 });

    document
        .querySelectorAll('.feature-card, .about-content, .research-stat')
        .forEach(el => observer.observe(el));
}

/* ================= MOBILE MENU ================= */
function setupMobileMenu() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');

    if (!hamburger || !navMenu) return;

    hamburger.addEventListener('click', () => {
        hamburger.classList.toggle('active');
        navMenu.classList.toggle('active');
    });

    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', () => {
            hamburger.classList.remove('active');
            navMenu.classList.remove('active');
        });
    });
}

/* ================= ANIMATIONS ================= */
function setupAnimations() {
    const brainIcon = document.querySelector('.brain-animation i');
    if (brainIcon) {
        setInterval(() => {
            brainIcon.style.transform = 'scale(1.1)';
            setTimeout(() => {
                brainIcon.style.transform = 'scale(1)';
            }, 200);
        }, 3000);
    }
}

/* ================= STATS (FIXED) ================= */
function setupStatsAnimation() {
    const stats = document.querySelectorAll('.stat-number');
    if (!stats.length) return;

    const animateStats = () => {
        stats.forEach(stat => {
            const rawText = stat.textContent.trim();

            // ✅ Skip non-numeric stats
            const number = parseFloat(rawText);
            if (isNaN(number)) return;

            const hasPercent = rawText.includes('%');
            let current = 0;
            const increment = number / 40;

            const timer = setInterval(() => {
                current += increment;
                if (current >= number) {
                    stat.textContent = number + (hasPercent ? '%' : '');
                    clearInterval(timer);
                } else {
                    stat.textContent = Math.floor(current) + (hasPercent ? '%' : '');
                }
            }, 30);
        });
    };

    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateStats();
                observer.disconnect();
            }
        });
    });

    const heroStats = document.querySelector('.hero-stats');
    if (heroStats) observer.observe(heroStats);
}

/* ================= CONTACT FORM ================= */
function handleContactForm(e) {
    e.preventDefault();

    const btn = e.target.querySelector('button[type="submit"]');
    if (!btn) return;

    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
    btn.disabled = true;

    setTimeout(() => {
        btn.innerHTML = '<i class="fas fa-check"></i> Message Sent!';
        btn.style.background = '#10b981';

        setTimeout(() => {
            btn.innerHTML = originalText;
            btn.disabled = false;
            btn.style.background = '';
            e.target.reset();
        }, 2000);
    }, 1500);
}

const helpModal = new bootstrap.Modal(document.getElementById('helpMessage'));
const onboardingSteps = require('./static/onboarding_steps');

class OnboardingController {
  constructor(steps = onboardingSteps) {
    this.steps = steps || [];
    this.showingId = null;
  }

  hasBeenShown(id) {
    return localStorage.get(`onboarding.${id}`, false);
  }

  markAsRead() {
    localStorage.set(`onboarding.${this.showingId}`, true);
    this.hideModal();
    this.showingId = null;
  }

  isValidStep(step, ignoreCondition = false) {
    return !this.hasBeenShown(step.id) && (step.condition() || ignoreCondition);
  }

  showAllTips() {
    for (const step of this.steps) {
      if (this.isValidStep(step)) {
        this.showModal(step.description);
        this.showingId = step.id;
        break;
      }
    }
  }

  showModal(content) {
    helpModal.show();
    document.getElementById('helpMessageContent').innerHTML = content;
  }

  hideModal() {
    document.getElementById('helpMessageContent').innerHTML = '';
    helpModal.hide();
  }

  showSpecificTips(ids) {
    for (const id of ids) {
      const step = this.steps.find((step) => step.id === id);
      if (step && this.isValidStep(step, true)) {
        this.showingId = step.id;
        chatController.chat.addFrontendMessage('onboarding', step.description);
        this.markAsRead();
      }
    }
  }
}

module.exports = OnboardingController;

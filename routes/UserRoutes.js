const express = require('express');
const router = express.Router()
const userController = require('../controllers/UserController');
const userAnalysisController = require('../controllers/userAnalysisController');

// Registration
router.post('/register',userController.registerUser);

// login
router.post('/login',userController.login);

// delete user
router.delete('/:id',userController.deleteUser);

// get all user
router.get('/',userController.getAllUsers);

//get user profile
router.get('/:id',userController.getUserDetail);

//update user profile
router.patch('/:id',userController.updateUserDetail);

// Add feedback to a user (typically for waiters)
router.post('/:userId/feedback', userController.addFeedback);

router.get('/:userId/feedback-analysis', userAnalysisController.getUserFeedbackAnalysis);
router.get('/orders/:id', userController.getOrders)

module.exports = router

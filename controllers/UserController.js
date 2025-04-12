const userServices = require('../services/UserServices');
const asyncErrorHandler = require('../middleware/asyncErrorHandler');
const User = require('../models/userModel');
const orderModel = require('../models/orderModel');


// Register User
exports.registerUser = asyncErrorHandler(async (req, res) => {  
     const newUser = await userServices.register(req.body);
    res.status(201).json({
        success: true,
        data: {
            user: newUser
        },
    });
});

// lgoin
exports.login = asyncErrorHandler(async (req, res) => {  
     const userdata = await userServices.login(req.body);
    res.status(201).json({
        success: true,
        user: userdata
       
    });
});


// Register User
exports.getAllUsers = asyncErrorHandler(async (req, res) => {  
     const cusromers = await userServices.getAllUsers();
    res.status(200).json({
        success: true,
        data: cusromers
    });
});



//delete user

exports.deleteUser=asyncErrorHandler(async(req,res)=>{
    const result = await userServices.delete(req.params.id);

    res.status(204).json({
        success: true,
    });

})



// user detail
exports.getUserDetail=asyncErrorHandler(async(req,res)=>{
    const user = await userServices.getUserDetail(req.params.id);
    res.status(200).json({
        success: true,
        data: user
    });

})


// update user detail
exports.updateUserDetail=asyncErrorHandler(async(req,res)=>{
    const user = await userServices.updateUserDetail(req.params.id,req.body);
    res.status(200).json({
        success: true,
        data: user
    });

})



//Feedback
exports.addFeedback = async (req, res) => {
    try {
        const { userId } = req.params;
        const { fromUserId, comment, rating } = req.body;

        // Validate input
        if (!fromUserId || !rating) {
            return res.status(400).json({
                success: false,
                message: 'fromUserId and rating are required'
            });
        }

        if (rating < 1 || rating > 5) {
            return res.status(400).json({
                success: false,
                message: 'Rating must be between 1 and 5'
            });
        }

        // Find the user to receive feedback
        const user = await User.findById(userId);
        if (!user) {
            return res.status(404).json({
                success: false,
                message: 'User not found'
            });
        }

        // Verify the user is a waiter (or other role that can receive feedback)
        if (!['waiter', 'chef'].includes(user.role)) {
            return res.status(400).json({
                success: false,
                message: 'Feedback can only be given to waiters or chefs'
            });
        }

        // Create new feedback
        const newFeedback = {
            fromUserId,
            comment: comment || '',
            rating,
            createdAt: new Date()
        };

        // Add feedback to the user
        user.feedback.push(newFeedback);

        // Calculate new average rating
        const totalRatings = user.feedback.reduce((sum, fb) => sum + fb.rating, 0);
        user.averageRating = totalRatings / user.feedback.length;

        // Save the updated user
        await user.save();

        res.status(201).json({
            success: true,
            message: 'Feedback added successfully',
            data: newFeedback
        });

    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: 'Server error',
            error: error.message
        });
    }
};

exports.getOrders = async(req,res) =>{
    try {
        const {id} = req.params
        const orders = await orderModel.find({user: id})
        res.status(200).json({
            orders
        })
    } catch (error) {
        console.log(error.message)
        res.status(400).json({msg: "an error has occured"})
    }
}

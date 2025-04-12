const User = require('../models/userModel');
const Order = require('../models/orderModel');
const customError = require('../utils/CustomErrorhandlerClass');

exports.register = async (userData) => {
    const { phoneNumber, email } = userData;
  
    // Validate required fields
    if (!phoneNumber) {
      throw new customError("Phone number is required.", 400);
    }
  
    // Check for duplicate phone number
    const existingUserWithPhone = await User.findOne({ phoneNumber });
    if (existingUserWithPhone) {
      throw new customError("Customer with this phone number already exists.", 400);
    }
  
    // Check for duplicate email (if provided and not empty)
    if (email && email.trim() !== "") {
      const existingUserWithEmail = await User.findOne({ email });
      if (existingUserWithEmail) {
        throw new customError("Customer with this email already exists.", 400);
      }
    }
  
    // If email is empty, omit it entirely
    if (email && email.trim() === "") {
      delete userData.email; // Remove the email field entirely
    }
  
    // Create the new user
    const user = new User(userData);
    await user.save();
  
    return user;
  };

  exports.login = async (userData) => {
    const { password, email } = userData;
  
    // Check for duplicate phone number
    const user = await User.findOne({ email });
    if (!user) {
      throw new customError("Customer with this email doesn't exists.", 400);
    }

    if(user.password != password){
      throw new customError("Incorrect password", 401);
    }
    return user;
  };

// get all User
exports.getAllUsers = async () => {  
    const customers = await User.find().sort({ createdAt: -1 }); ;
    if (!customers || customers.length === 0) {
        throw new customError("Customers doesn't exist.", 400);
    }
    return customers;  
};

// get User detail and order history
exports.getUserDetail = async (userId) => {  
    const customer = await User.findById({user: userId});
    if (!customer) {
        throw new customError("Customer doesn't exist.", 400);
    }

    // Fetch the customer's order history
    const orders = await Order.find({ customer: userId }).populate('items.foodItem');
    
    return { customer, orders };
};





exports.updateUserDetail = async (userId, userData) => {
    // Fetch the current user's data
    const currentUser = await User.findById(userId);
  
    if (!currentUser) {
      throw new customError("Customer doesn't exist.", 400);
    }
  
    // Check if email is being updated to a non-empty value
    if (userData.email && userData.email.trim() !== "") {
      // Skip duplicate check if the email is the same as the current user's email
      if (userData.email !== currentUser.email) {
        const existingUser = await User.findOne({
          email: userData.email,
          _id: { $ne: userId }, // Exclude current user
        });
  
        if (existingUser) {
          throw new customError("Customer with this email already exists.", 400);
        }
      }
    }
  
    // Check if phoneNumber is being updated
    if (userData.phoneNumber) {
      // Skip duplicate check if the phone number is the same as the current user's phone number
      if (userData.phoneNumber !== currentUser.phoneNumber) {
        const existingUser = await User.findOne({
          phoneNumber: userData.phoneNumber,
          _id: { $ne: userId }, // Exclude current user
        });
  
        if (existingUser) {
          throw new customError("Customer with this phone number already exists.", 400);
        }
      }
    }
  
// If email is being removed (omitted from the request), set it to null
  if (userData.email === undefined) {
    userData.$unset = { email: "" };
  }
  
    // Proceed with update
    const customer = await User.findByIdAndUpdate(userId, userData, { new: true });
  
    return customer;
  };


// delete User
exports.delete = async (userId) => {
 
    const existingUser = await User.findById(userId);
    if (!existingUser) {
        throw new customError("Customer doesn't exist.", 400);
    }
    
    await existingUser.deleteOne()

};

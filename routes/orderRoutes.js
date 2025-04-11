const express = require("express");
const orderController = require("../controllers/orderController");
const Order = require("../models/orderModel");
const axios = require("axios");
const FoodItem = require("../models/foodItem");
const { default: mongoose } = require("mongoose");
const { HfInference } = require('@huggingface/inference');
const User = require("../models/userModel");

const router = express.Router();
const hf = new HfInference(process.env.HUGGINGFACE_API_KEY);

// Place a new order
router.post("/", orderController.createOrder);

// Get recent orders (for home screen)
router.get("/recent", orderController.getRecentOrders);

// Enhanced AI Recommendation Engine with Full Feedback Analysis
async function generateChefRecommendation(userId, orderedItems) {
    try {
        // Get ALL feedback from this user across all food items with proper population
        const foodItemsWithFeedback = await FoodItem.find({
            'feedback.userId': userId
        })
        .populate({
            path: 'feedback.userId',
            select: 'firstName lastName',
            model: 'User'
        })
        .lean();

        // Extract and format all user feedback
        const allUserFeedback = foodItemsWithFeedback.flatMap(foodItem => 
            foodItem.feedback
                .filter(fb => fb.userId && fb.userId._id.toString() === userId.toString())
                .map(fb => ({
                    foodItem: foodItem.name,
                    comment: fb.comment,
                    rating: fb.rating,
                    createdAt: fb.createdAt,
                    foodItemId: foodItem._id
                }))
        );

        if (allUserFeedback.length === 0) {
            return {
                recommendation: "No previous feedback found from this customer. Prepare using standard recipes and presentation.",
                priority: "normal",
                sentiment: "neutral",
                feedbackHistory: []
            };
        }

        // Step 1: Sentiment Analysis for each feedback comment
        const feedbackWithSentiment = await Promise.all(allUserFeedback.map(async (fb) => {
            try {
                const sentimentResult = await hf.textClassification({
                    model: 'cardiffnlp/twitter-roberta-base-sentiment',
                    inputs: fb.comment
                });
                
                const topSentiment = sentimentResult[0];
                return {
                    ...fb,
                    sentiment: topSentiment.label,
                    sentimentScore: topSentiment.score
                };
            } catch (error) {
                console.error(`Error analyzing sentiment for comment: "${fb.comment}"`, error);
                return {
                    ...fb,
                    sentiment: 'neutral',
                    sentimentScore: 0.5
                };
            }
        }));

        // Step 2: Summarize feedback themes using BART
        const feedbackSummaryText = feedbackWithSentiment.map(fb => 
            `On ${new Date(fb.createdAt).toLocaleDateString()} for ${fb.foodItem}: Rated ${fb.rating}/5 - "${fb.comment}" (${fb.sentiment}, ${(fb.sentimentScore * 100).toFixed(1)}% confidence)`
        ).join('\n');

        let feedbackSummary;
        try {
            const summaryResponse = await hf.summarization({
                model: 'facebook/bart-large-cnn',
                inputs: feedbackSummaryText,
                parameters: {
                    max_length: 150,
                    min_length: 50
                }
            });
            feedbackSummary = summaryResponse.summary_text;
        } catch (error) {
            console.error("Error summarizing feedback:", error);
            feedbackSummary = "Customer has provided mixed feedback across various dishes. Check individual comments for details.";
        }

        // Step 3: Generate specific recommendations based on feedback and current order
        const currentOrderItems = orderedItems.map(item => 
            `${item.quantity}x ${item.foodItemId?.name || item.name}`
        ).join(', ');

        let recommendation;
        try {
            const prompt = `As a professional chef, analyze this customer's feedback history and current order to generate specific cooking recommendations.

CUSTOMER FEEDBACK SUMMARY:
${feedbackSummary}

CURRENT ORDER ITEMS:
${currentOrderItems}

Provide detailed technical suggestions including:
- Seasoning adjustments
- Cooking techniques
- Presentation tips
- Quality control measures
- Any special considerations based on their preferences`;

            const recommendationResponse = await hf.textGeneration({
                model: 'tiiuae/falcon-7b-instruct',
                inputs: prompt,
                parameters: {
                    max_new_tokens: 500,
                    temperature: 0.7,
                    return_full_text: false
                }
            });
            recommendation = recommendationResponse.generated_text.trim();
        } catch (error) {
            console.error("Error generating recommendations:", error);
            recommendation = "Standard preparation recommended. Check customer's past feedback for potential preferences.";
        }

        // Calculate priority based on historical ratings and sentiment
        const avgRating = allUserFeedback.reduce((sum, fb) => sum + fb.rating, 0) / allUserFeedback.length;
        const negativeFeedbackCount = feedbackWithSentiment.filter(fb => fb.sentiment === 'negative').length;
        
        let priority = "normal";
        if (avgRating < 2.5 || negativeFeedbackCount > allUserFeedback.length * 0.5) priority = "high";
        else if (avgRating > 4.2 && negativeFeedbackCount === 0) priority = "low";

        return {
            recommendation,
            priority,
            sentiment: avgRating >= 3 ? "positive" : avgRating >= 2 ? "neutral" : "negative",
            analyzedAt: new Date(),
            historicalData: {
                averageRating: parseFloat(avgRating.toFixed(1)),
                totalFeedback: allUserFeedback.length,
                lastFeedbackDate: new Date(allUserFeedback[allUserFeedback.length - 1].createdAt).toLocaleDateString(),
                negativeFeedbackCount,
                positiveFeedbackCount: feedbackWithSentiment.filter(fb => fb.sentiment === 'positive').length
            },
            feedbackSamples: feedbackWithSentiment.slice(0, 3).map(fb => ({
                foodItem: fb.foodItem,
                comment: fb.comment,
                rating: fb.rating,
                sentiment: fb.sentiment,
                date: new Date(fb.createdAt).toLocaleDateString()
            }))
        };

    } catch (error) {
        console.error("AI recommendation error:", error);
        return {
            recommendation: "Standard preparation required. Check customer's past ratings for potential preferences.",
            priority: "normal",
            sentiment: "neutral",
            analyzedAt: new Date(),
            feedbackHistory: []
        };
    }
}

// Get orders for chef with comprehensive AI analysis
router.get("/chef", async (req, res) => {
    try {
        const { analyze } = req.query;
        const statusFilter = ["pending", "preparing"];

        // Fetch orders with full population
        let orders = await Order.find({ status: { $in: statusFilter } })
            .sort({ createdAt: 1 })
            .populate({
                path: "user",
                select: "firstName lastName role",
                model: "User"
            })
            .populate({
                path: "items.foodItemId",
                select: "name price description preparationTime",
                model: "FoodItem"
            })
            .lean();

        // Format order data
        orders = orders.map(order => {
            const orderNumber = order._id.toString().substring(18, 24).toUpperCase();
            const orderTime = new Date(order.createdAt);
            const now = new Date();
            const diffMinutes = Math.floor((now - orderTime) / (1000 * 60));
            
            return {
                ...order,
                orderNumber: `#${orderNumber}`,
                orderTime: `${Math.floor(diffMinutes/60)}h ${diffMinutes%60}m ago`,
                status: order.status.toUpperCase(),
                formattedItems: order.items.map(item => ({
                    name: item.foodItemId?.name || item.name,
                    quantity: item.quantity,
                    price: item.price || item.foodItemId?.price,
                    preparationTime: item.foodItemId?.preparationTime || 15,
                    specialInstructions: item.specialInstructions || ''
                })),
                customerName: order.user ? `${order.user.firstName} ${order.user.lastName}` : 'Guest'
            };
        });

        // Add AI analysis if requested
        if (analyze === "true") {
            orders = await Promise.all(orders.map(async (order) => {
                if (!order.user) {
                    return {
                        ...order,
                        analysis: {
                            recommendation: "Guest user - no feedback history available",
                            priority: "normal",
                            sentiment: "neutral"
                        }
                    };
                }

                const analysis = await generateChefRecommendation(
                    order.user._id, 
                    order.items
                );

                return {
                    ...order,
                    analysis
                };
            }));
        }

        res.status(200).json(orders);

    } catch (error) {
        console.error("Error in chef orders route:", error);
        res.status(500).json({
            message: "Failed to process orders",
            error: error.message,
            stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
        });
    }
});

// Update order status
router.patch("/:id/status", async (req, res) => {
    try {
        const { status } = req.body;
        const validStatuses = ["pending", "preparing", "ready", "completed", "cancelled"];

        if (!validStatuses.includes(status)) {
            return res.status(400).json({ message: "Invalid status value" });
        }

        const order = await Order.findByIdAndUpdate(
            req.params.id,
            { status, updatedAt: new Date() },
            { new: true }
        );

        if (!order) {
            return res.status(404).json({ message: "Order not found" });
        }

        res.status(200).json(order);
    } catch (error) {
        console.error("Error updating order status:", error);
        res.status(500).json({
            message: "Failed to update order",
            error: error.message,
        });
    }
});

// Utility function for sentiment analysis
async function analyzeSentiment(text) {
    if (!text || text.trim().length === 0) return { label: 'neutral', score: 0.5 };
    
    try {
        const result = await hf.textClassification({
            model: 'cardiffnlp/twitter-roberta-base-sentiment',
            inputs: text
        });
        return {
            label: result[0].label,
            score: result[0].score
        };
    } catch (error) {
        console.error("Sentiment analysis error:", error);
        return {
            label: 'neutral',
            score: 0.5
        };
    }
}

// 1. Overall Customer Satisfaction Dashboard
router.get("/analytics/satisfaction", async (req, res) => {
    try {
        // Get all food items with feedback
        const foodItems = await FoodItem.find({ 'feedback.0': { $exists: true } })
            .populate('feedback.userId', 'firstName lastName');

        // Get all completed orders
        const orders = await Order.find({ status: 'completed' })
            .populate('user', 'firstName lastName');

        // Calculate overall ratings
        const allFeedback = foodItems.flatMap(item => item.feedback);
        const totalReviews = allFeedback.length;
        const averageRating = totalReviews > 0 
            ? allFeedback.reduce((sum, fb) => sum + fb.rating, 0) / totalReviews
            : 0;

        // Sentiment analysis
        const sentimentResults = await Promise.all(
            allFeedback.map(fb => analyzeSentiment(fb.comment))
        );
        const sentimentDistribution = sentimentResults.reduce((acc, { label }) => {
            acc[label] = (acc[label] || 0) + 1;
            return acc;
        }, { positive: 0, neutral: 0, negative: 0 });

        // Recent feedback samples
        const recentFeedback = allFeedback
            .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
            .slice(0, 5)
            .map(fb => ({
                _id: fb._id,
                foodItem: foodItems.find(item => item.feedback.some(f => f._id.equals(fb._id)))?.name || 'Unknown',
                user: fb.userId?.firstName || 'Anonymous',
                rating: fb.rating,
                comment: fb.comment,
                sentiment: sentimentResults.find(sr => sr.input === fb.comment)?.label || 'neutral',
                date: fb.createdAt,
                replied: fb.reply ? true : false,
                reply: fb.reply || null
            }));

        // Customer retention metrics
        const repeatCustomers = await Order.aggregate([
            { $match: { status: 'completed' } },
            { $group: { _id: '$user', count: { $sum: 1 } } },
            { $match: { count: { $gt: 1 } } },
            { $count: 'repeatCustomers' }
        ]);

        res.status(200).json({
            success: true,
            metrics: {
                totalCustomers: await User.countDocuments(),
                activeCustomers: await Order.distinct('user').countDocuments(),
                repeatCustomers: repeatCustomers[0]?.repeatCustomers || 0,
                totalReviews,
                averageRating: parseFloat(averageRating.toFixed(1)),
                sentimentDistribution,
                satisfactionScore: calculateSatisfactionScore(averageRating, sentimentDistribution),
                trendingItems: await getTrendingItems(),
                issuesReported: await countReportedIssues()
            },
            recentFeedback,
            timestamp: new Date()
        });

    } catch (error) {
        console.error("Admin analytics error:", error);
        res.status(500).json({
            success: false,
            message: "Failed to generate satisfaction analytics",
            error: process.env.NODE_ENV === 'development' ? error.message : null
        });
    }
});

// 2. Customer Satisfaction Over Time
router.get("/analytics/satisfaction-trend", async (req, res) => {
    try {
        const { period = 'month' } = req.query; // day, week, month, year
        
        const foodItems = await FoodItem.aggregate([
            { $unwind: '$feedback' },
            {
                $group: {
                    _id: {
                        [getDateGrouping(period)]: '$feedback.createdAt'
                    },
                    averageRating: { $avg: '$feedback.rating' },
                    count: { $sum: 1 }
                }
            },
            { $sort: { '_id': 1 } },
            { $limit: 30 }
        ]);

        // Add sentiment data per period
        const resultsWithSentiment = await Promise.all(
            foodItems.map(async periodData => {
                const feedbackInPeriod = await FoodItem.aggregate([
                    { $unwind: '$feedback' },
                    {
                        $match: {
                            'feedback.createdAt': {
                                $gte: new Date(periodData._id),
                                $lt: getNextPeriodDate(new Date(periodData._id), period)
                            }
                        }
                    },
                    { $sample: { size: 10 } }
                ]);

                const sentimentAnalysis = await Promise.all(
                    feedbackInPeriod.map(fb => analyzeSentiment(fb.feedback.comment))
                );

                const sentimentCount = sentimentAnalysis.reduce((acc, { label }) => {
                    acc[label] = (acc[label] || 0) + 1;
                    return acc;
                }, { positive: 0, neutral: 0, negative: 0 });

                return {
                    ...periodData,
                    sentiment: sentimentCount
                };
            })
        );

        res.status(200).json({
            success: true,
            period,
            data: resultsWithSentiment,
            timestamp: new Date()
        });

    } catch (error) {
        console.error("Satisfaction trend error:", error);
        res.status(500).json({
            success: false,
            message: "Failed to generate satisfaction trend",
            error: process.env.NODE_ENV === 'development' ? error.message : null
        });
    }
});

// 3. Individual Customer Satisfaction Profiles
router.get("/analytics/customers/:id/satisfaction", async (req, res) => {
    try {
        const { id } = req.params;

        if (!mongoose.Types.ObjectId.isValid(id)) {
            return res.status(400).json({ 
                success: false,
                message: "Invalid customer ID" 
            });
        }

        // Get all feedback from this customer
        const foodItems = await FoodItem.find({ 'feedback.userId': id })
            .populate('feedback.userId', 'firstName lastName');

        const customerFeedback = foodItems.flatMap(item => 
            item.feedback.filter(fb => fb.userId && fb.userId._id.toString() === id)
        );

        if (customerFeedback.length === 0) {
            return res.status(200).json({
                success: true,
                message: "No feedback found for this customer",
                customerId: id,
                hasFeedback: false
            });
        }

        // Calculate customer stats
        const averageRating = customerFeedback.reduce((sum, fb) => sum + fb.rating, 0) / customerFeedback.length;
        const sentimentResults = await Promise.all(
            customerFeedback.map(fb => analyzeSentiment(fb.comment))
        );
        const sentimentDistribution = sentimentResults.reduce((acc, { label }) => {
            acc[label] = (acc[label] || 0) + 1;
            return acc;
        }, { positive: 0, neutral: 0, negative: 0 });

        // Get customer's order history
        const orderHistory = await Order.find({ user: id, status: 'completed' })
            .sort({ createdAt: -1 })
            .limit(10)
            .populate('items.foodItemId', 'name');

        res.status(200).json({
            success: true,
            customerId: id,
            customerName: customerFeedback[0].userId?.firstName + ' ' + customerFeedback[0].userId?.lastName,
            metrics: {
                totalFeedback: customerFeedback.length,
                averageRating: parseFloat(averageRating.toFixed(1)),
                sentimentDistribution,
                satisfactionScore: calculateSatisfactionScore(averageRating, sentimentDistribution),
                lastOrderDate: orderHistory[0]?.createdAt || null,
                totalOrders: await Order.countDocuments({ user: id })
            },
            recentFeedback: customerFeedback
                .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
                .slice(0, 5)
                .map(fb => ({
                    foodItem: foodItems.find(item => item.feedback.some(f => f._id.equals(fb._id)))?.name || 'Unknown',
                    rating: fb.rating,
                    comment: fb.comment,
                    sentiment: sentimentResults.find(sr => sr.input === fb.comment)?.label || 'neutral',
                    date: fb.createdAt
                })),
            orderHistory: orderHistory.map(order => ({
                orderId: order._id,
                date: order.createdAt,
                items: order.items.map(item => item.foodItemId?.name || 'Unknown'),
                total: order.totalAmount
            })),
            timestamp: new Date()
        });

    } catch (error) {
        console.error("Customer satisfaction profile error:", error);
        res.status(500).json({
            success: false,
            message: "Failed to generate customer satisfaction profile",
            error: process.env.NODE_ENV === 'development' ? error.message : null
        });
    }
});

// 4. Food Item Satisfaction Analysis
router.get("/analytics/food-items/:id/satisfaction", async (req, res) => {
    try {
        const { id } = req.params;

        if (!mongoose.Types.ObjectId.isValid(id)) {
            return res.status(400).json({ 
                success: false,
                message: "Invalid food item ID" 
            });
        }

        const foodItem = await FoodItem.findById(id)
            .populate('feedback.userId', 'firstName lastName');

        if (!foodItem) {
            return res.status(404).json({ 
                success: false,
                message: "Food item not found" 
            });
        }

        if (foodItem.feedback.length === 0) {
            return res.status(200).json({
                success: true,
                message: "No feedback found for this item",
                foodItemId: id,
                foodItemName: foodItem.name,
                hasFeedback: false
            });
        }

        // Calculate item stats
        const averageRating = foodItem.feedback.reduce((sum, fb) => sum + fb.rating, 0) / foodItem.feedback.length;
        const sentimentResults = await Promise.all(
            foodItem.feedback.map(fb => analyzeSentiment(fb.comment))
        );
        const sentimentDistribution = sentimentResults.reduce((acc, { label }) => {
            acc[label] = (acc[label] || 0) + 1;
            return acc;
        }, { positive: 0, neutral: 0, negative: 0 });

        // Get recent orders containing this item
        const recentOrders = await Order.aggregate([
            { $unwind: '$items' },
            { $match: { 'items.foodItemId': mongoose.Types.ObjectId(id) } },
            { $sort: { createdAt: -1 } },
            { $limit: 10 },
            {
                $lookup: {
                    from: 'users',
                    localField: 'user',
                    foreignField: '_id',
                    as: 'user'
                }
            },
            { $unwind: '$user' }
        ]);

        res.status(200).json({
            success: true,
            foodItemId: id,
            foodItemName: foodItem.name,
            metrics: {
                totalFeedback: foodItem.feedback.length,
                averageRating: parseFloat(averageRating.toFixed(1)),
                sentimentDistribution,
                satisfactionScore: calculateSatisfactionScore(averageRating, sentimentDistribution),
                lastOrdered: recentOrders[0]?.createdAt || null,
                totalOrders: await Order.countDocuments({ 'items.foodItemId': id })
            },
            recentFeedback: foodItem.feedback
                .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
                .slice(0, 5)
                .map((fb, index) => ({
                    customer: fb.userId?.firstName + ' ' + fb.userId?.lastName || 'Anonymous',
                    rating: fb.rating,
                    comment: fb.comment,
                    sentiment: sentimentResults[index]?.label || 'neutral',
                    date: fb.createdAt,
                    reply: fb.reply || null
                })),
            recentOrders: recentOrders.map(order => ({
                orderId: order._id,
                date: order.createdAt,
                customer: order.user.firstName + ' ' + order.user.lastName,
                quantity: order.items.find(i => i.foodItemId.toString() === id).quantity
            })),
            timestamp: new Date()
        });

    } catch (error) {
        console.error("Food item satisfaction error:", error);
        res.status(500).json({
            success: false,
            message: "Failed to generate food item satisfaction analysis",
            error: process.env.NODE_ENV === 'development' ? error.message : null
        });
    }
});

// 5. Add reply to feedback
router.post("/food-items/:foodItemId/feedback/:feedbackId/reply", async (req, res) => {
    try {
        const { foodItemId, feedbackId } = req.params;
        const { reply } = req.body;

        if (!mongoose.Types.ObjectId.isValid(foodItemId) || !mongoose.Types.ObjectId.isValid(feedbackId)) {
            return res.status(400).json({ 
                success: false,
                message: "Invalid food item or feedback ID" 
            });
        }

        const foodItem = await FoodItem.findById(foodItemId);
        if (!foodItem) {
            return res.status(404).json({ 
                success: false,
                message: "Food item not found" 
            });
        }

        const feedback = foodItem.feedback.id(feedbackId);
        if (!feedback) {
            return res.status(404).json({ 
                success: false,
                message: "Feedback not found" 
            });
        }

        feedback.reply = reply;
        feedback.repliedAt = new Date();
        await foodItem.save();

        res.status(200).json({
            success: true,
            message: "Reply added successfully",
            feedback: {
                _id: feedback._id,
                comment: feedback.comment,
                rating: feedback.rating,
                reply: feedback.reply,
                repliedAt: feedback.repliedAt
            }
        });

    } catch (error) {
        console.error("Error adding reply to feedback:", error);
        res.status(500).json({
            success: false,
            message: "Failed to add reply to feedback",
            error: process.env.NODE_ENV === 'development' ? error.message : null
        });
    }
});

// Helper Functions

function calculateSatisfactionScore(averageRating, sentimentDistribution) {
    const ratingWeight = 0.6;
    const sentimentWeight = 0.4;
    
    const maxSentiment = Math.max(
        sentimentDistribution.positive || 0,
        sentimentDistribution.neutral || 0,
        sentimentDistribution.negative || 0
    );
    
    const sentimentRatio = maxSentiment > 0 
        ? (sentimentDistribution.positive || 0) / maxSentiment
        : 0.5;
    
    return parseFloat((
        (averageRating / 5 * ratingWeight) + 
        (sentimentRatio * sentimentWeight)
    ).toFixed(2)) * 100;
}

function getDateGrouping(period) {
    switch (period) {
        case 'day': return '$dayOfYear';
        case 'week': return '$week';
        case 'month': return '$month';
        case 'year': return '$year';
        default: return '$month';
    }
}

function getNextPeriodDate(date, period) {
    const result = new Date(date);
    switch (period) {
        case 'day': result.setDate(result.getDate() + 1); break;
        case 'week': result.setDate(result.getDate() + 7); break;
        case 'month': result.setMonth(result.getMonth() + 1); break;
        case 'year': result.setFullYear(result.getFullYear() + 1); break;
    }
    return result;
}

async function getTrendingItems() {
    const lastWeek = new Date();
    lastWeek.setDate(lastWeek.getDate() - 7);
    
    return FoodItem.aggregate([
        {
            $lookup: {
                from: 'orders',
                localField: '_id',
                foreignField: 'items.foodItemId',
                as: 'orders'
            }
        },
        { 
            $project: {
                name: 1,
                totalOrders: { $size: '$orders' },
                recentOrders: {
                    $size: {
                        $filter: {
                            input: '$orders',
                            as: 'order',
                            cond: { $gte: ['$$order.createdAt', lastWeek] }
                        }
                    }
                },
                rating: 1
            }
        },
        { $sort: { recentOrders: -1, rating: -1 } },
        { $limit: 5 }
    ]);
}

async function countReportedIssues() {
    return FoodItem.aggregate([
        { $unwind: '$feedback' },
        {
            $match: {
                $or: [
                    { 'feedback.rating': { $lt: 3 } },
                    { 
                        'feedback.comment': { 
                            $regex: /(bad|poor|terrible|horrible|awful|disappointing|issue|problem)/i 
                        } 
                    }
                ]
            }
        },
        {
            $group: {
                _id: null,
                count: { $sum: 1 },
                unresolved: {
                    $sum: {
                        $cond: [{ $eq: ['$feedback.resolved', false] }, 1, 0]
                    }
                }
            }
        }
    ]).then(results => results[0] || { count: 0, unresolved: 0 });
}

module.exports = router;
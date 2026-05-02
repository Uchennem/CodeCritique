import 'dotenv/config';
import express from 'express';
import indexRoutes from './routes/index.js';

const app = express();
const port = process.env.PORT || 3000;

// middleware
app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(express.static('public'));

// view engine
app.set('view engine', 'ejs');

// routes
app.use('/', indexRoutes);

app.listen(port, () => {
  console.log(`Server running on port: ${port}`);
});

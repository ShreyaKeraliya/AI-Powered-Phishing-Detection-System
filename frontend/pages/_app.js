import Footer from "../components/Footer";
import Navbar from "../components/NavBar";
import "../styles/globals.css";

export default function App({ Component, pageProps }) {
  return <>
  <Navbar/>
  <Component {...pageProps} />
  <Footer/>
  </>
}


